import json
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional

import torch

from DLMUSE.utils import prepare_data_folder, rename_and_copy_files

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def run_dlmuse_pipeline(
    in_dir: str,
    out_dir: str,
    device: str,
    verbose: bool = False,
    save_probabilities: bool = False,
    continue_prediction: bool = False,
    disable_progress_bar: bool = False,
    clear_cache: bool = False,
    disable_tta: bool = False,
    d: str = "903",
    p: str = "nnUNetPlans",
    tr: str = "nnUNetTrainer",
    c: str = "3d_fullres",
    f: list = [0],
    step_size: float = 0.5,
    chk: str = "checkpoint_final.pth",
    npp: int = 2,
    nps: int = 2,
    prev_stage_predictions: Optional[str] = None,
    num_parts: int = 1,
    part_id: int = 0,
) -> None:

    if clear_cache:
        shutil.rmtree(os.path.join(Path(__file__).parent, "nnunet_results"))
        shutil.rmtree(os.path.join(Path(__file__).parent, ".cache"))
        if not in_dir or not out_dir:
            logging.error("Cache cleared and missing either -i / -o. Exiting.")
            sys.exit(0)

    if not in_dir or not out_dir:
        logging.error("The following arguments are required: -i, -o")
        sys.exit(0)

    # data conversion
    src_folder = in_dir  # input folder

    if not os.path.exists(out_dir):  # create output folder if it does not exist
        logging.info(f"Can't find {out_dir}, creating it...")
        os.makedirs(out_dir)

    des_folder = os.path.join(out_dir, "renamed_image")

    # check if -i argument is a folder, list (csv), or a single file (nii.gz)
    if os.path.isdir(in_dir):  # if in_dir is a directory
        src_folder = in_dir
        prepare_data_folder(des_folder)
        rename_dic, rename_back_dict = rename_and_copy_files(src_folder, des_folder)
        datalist_file = os.path.join(des_folder, "renaming.json")
        with open(datalist_file, "w", encoding="utf-8") as _f:
            json.dump(rename_dic, _f, ensure_ascii=False, indent=4)
        logging.info(f"Renaming dic is saved to {datalist_file}")
    else:
        logging.error("Input directory not found. Exiting DLMUSE.")
        sys.exit(0)

    model_folder = os.path.join(
        Path(__file__).parent,
        "nnunet_results",
        "Dataset%s_Task%s_DLMUSEV2/nnUNetTrainer__nnUNetPlans__%s/" % (d, d, c),
    )

    # Check if model exists. If not exist, download using HuggingFace
    logging.info(f"Using model folder: {model_folder}")
    if not os.path.exists(model_folder):
        # HF download model
        logging.info("DLMUSE model not found, downloading...")

        from huggingface_hub import snapshot_download

        local_src = Path(__file__).parent
        snapshot_download(repo_id="nichart/DLMUSE", local_dir=local_src)

        logging.info("DLMUSE model has been successfully downloaded!")
    else:
        logging.info("Loading the model...")

    prepare_data_folder(des_folder)

    assert part_id < num_parts, "part_id < num_parts. Please see nnUNetv2_predict -h."

    assert device in [
        "cpu",
        "cuda",
        "mps",
    ], f"-device must be either cpu, mps or cuda. Got: {device}."

    if device == "cpu":
        import multiprocessing

        # use half of the available threads in the system.
        torch.set_num_threads(multiprocessing.cpu_count() // 2)
        device = torch.device("cpu")
        logging.info("Running in CPU mode.")
    elif device == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
        logging.info("Running in CUDA mode.")
    else:
        device = torch.device("mps")
        logging.info("Running in MPS mode.")

    # exports for nnunetv2 purposes
    os.environ["nnUNet_raw"] = "/nnunet_raw/"
    os.environ["nnUNet_preprocessed"] = "/nnunet_preprocessed"
    os.environ["nnUNet_results"] = (
        "/nnunet_results"  # where model will be located (fetched from HF)
    )

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    # Keep the outputs consistent
    torch.use_deterministic_algorithms(True)

    # Initialize nnUnetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=not disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=verbose,
        verbose_preprocessing=verbose,
        allow_tqdm=not disable_progress_bar,
    )

    # Retrieve the model and it's weight
    predictor.initialize_from_trained_model_folder(model_folder, f, checkpoint_name=chk)

    # Final prediction
    predictor.predict_from_files(
        des_folder,
        out_dir,
        save_probabilities=save_probabilities,
        overwrite=not continue_prediction,
        num_processes_preprocessing=npp,
        num_processes_segmentation_export=nps,
        folder_with_segs_from_prev_stage=prev_stage_predictions,
        num_parts=num_parts,
        part_id=part_id,
    )

    # After prediction, convert the image name back to original
    files_folder = out_dir

    for filename in os.listdir(files_folder):
        if filename.endswith(".nii.gz"):
            original_name = rename_back_dict[filename]
            os.rename(
                os.path.join(files_folder, filename),
                os.path.join(files_folder, original_name),
            )
    # Remove the (temporary) des_folder directory
    if os.path.exists(des_folder):
        shutil.rmtree(des_folder)

    logging.info("Inference Process Done!")
