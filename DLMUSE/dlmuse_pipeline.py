import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch

from .utils import prepare_data_folder, rename_and_copy_files


def run_pipeline(
    in_dir: str,
    out_dir: str,
    device: str,
    clear_cache: bool = False,
    d: str = "903",
    c: str = "3d_fullres",
    part_id: int = 0,
    num_parts: int = 1,
    step_size: float = 0.5,
    disable_tta: bool = False,
    verbose: bool = False,
    disable_progress_bar: bool = False,
    chk: bool = False,
    save_probabilities: bool = False,
    continue_prediction: bool = False,
    npp: int = 2,
    nps: int = 2,
    prev_stage_predictions: Optional[str] = None,
) -> None:
    """
    Run dlmuse pipeline function
    :param in_dir: The input directory
    :type in_dir: str
    :param out_dir: The output directory
    :type out_dir: str
    :param device: cpu/cuda/mps
    :type device: str

    Any other argument is not needed for 99% of you.
    Devs should see the code

    :rtype: None
    """
    f = [0]
    if clear_cache:
        shutil.rmtree(os.path.join(Path(__file__).parent, "nnunet_results"))
        shutil.rmtree(os.path.join(Path(__file__).parent, ".cache"))
        if not in_dir or not out_dir:
            print("Cache cleared and missing either -i / -o. Exiting.")
            sys.exit(0)

    if not in_dir or not out_dir:
        print("The following arguments are required: -i, -o")
        sys.exit(0)

    # data conversion
    src_folder = in_dir  # input folder
    if not os.path.exists(out_dir):  # create output folder if it does not exist
        os.makedirs(out_dir)

    des_folder = os.path.join(out_dir, "renamed_image")

    # check if -i argument is a folder, list (csv), or a single file (nii.gz)
    if os.path.isdir(in_dir):  # if args.i is a directory
        src_folder = in_dir
        prepare_data_folder(des_folder)
        rename_dic, rename_back_dict = rename_and_copy_files(src_folder, des_folder)
        datalist_file = os.path.join(des_folder, "renaming.json")
        with open(datalist_file, "w", encoding="utf-8") as ff:
            json.dump(rename_dic, ff, ensure_ascii=False, indent=4)
        print(f"Renaming dic is saved to {datalist_file}")

    model_folder = os.path.join(
        Path(__file__).parent,
        "nnunet_results",
        "Dataset%s_Task%s_DLMUSEV2/nnUNetTrainer__nnUNetPlans__%s/" % (d, d, c),
    )

    if clear_cache:
        shutil.rmtree(os.path.join(Path(__file__).parent, "nnunet_results"))
        shutil.rmtree(os.path.join(Path(__file__).parent, ".cache"))

    # Check if model exists. If not exist, download using HuggingFace
    if not os.path.exists(model_folder):
        # HF download model
        print("DLICV model not found, downloading...")

        from huggingface_hub import snapshot_download

        local_src = Path(__file__).parent
        snapshot_download(repo_id="nichart/DLMUSE", local_dir=local_src)
        print("DLMUSE model has been successfully downloaded!")
    else:
        print("Loading the model...")

    prepare_data_folder(out_dir)

    assert (
        part_id < num_parts
    ), "part_id < num_parts. Please see nnUNetv2_predict -h."

    assert device in [
        "cpu",
        "cuda",
        "mps",
    ], f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {device}."

    if device == "cpu":
        import multiprocessing

        torch.set_num_threads(
            multiprocessing.cpu_count() // 2
        )  # use half of the threads (better for PC)
        device = torch.device("cpu")
    elif device == "cuda":
        # multithreading in torch doesn't help nnU-Netv2 if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    # exports for nnunetv2 purposes
    os.environ["nnUNet_raw"] = "/nnunet_raw/"
    os.environ["nnUNet_preprocessed"] = "/nnunet_preprocessed"
    os.environ["nnUNet_results"] = (
        "/nnunet_results"  # where model will be located (fetched from HF)
    )

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

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

    # Retrieve the model and its weight
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

    print("DLICV Process Done!")
