import json
import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from DLMUSE.utils import prepare_data_folder, rename_and_copy_files

# nnU-Net / torch emit a lot of noisy warnings that are irrelevant to end users
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).parent
VALID_DEVICES = ("cpu", "cuda", "mps")


def _setup_device(device: str) -> torch.device:
    if device == "cpu":
        # use half of the available threads in the system
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    elif device == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    logger.info(f"Running in {device.upper()} mode.")
    return torch.device(device)


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
    f: Optional[List[int]] = None,
    step_size: float = 0.5,
    chk: str = "checkpoint_final.pth",
    npp: int = 2,
    nps: int = 2,
    prev_stage_predictions: Optional[str] = None,
    num_parts: int = 1,
    part_id: int = 0,
    low_memory: bool = False,
    qc: bool = False,
    qc_thresholds: Union[None, str, Dict[str, Any]] = None,
) -> None:
    """
    qc_thresholds: optional overrides for the QC thresholds, as a dict or JSON
    string containing only the keys to change, e.g. {"max_wrong_side": 0.2}.
    Valid keys: max_wrong_side, midline_tolerance, max_asymmetry, min_pairs,
    and QC_detail (1 = also write the per-ROI DLMUSE_QC_detail.csv).
    """
    folds = [0] if f is None else f

    if qc:
        from DLMUSE.qc import parse_qc_thresholds

        # Validate QC thresholds up front so a bad value doesn't fail after inference
        qc_thresholds = parse_qc_thresholds(qc_thresholds)

    # Validate arguments before doing any work
    if device not in VALID_DEVICES:
        raise ValueError(f"-device must be one of {VALID_DEVICES}. Got: {device}.")
    if part_id >= num_parts:
        raise ValueError(
            "part_id must be < num_parts. Please see nnUNetv2_predict -h."
        )
    if not in_dir or not out_dir:
        raise ValueError("The following arguments are required: -i, -o")
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    if clear_cache:
        shutil.rmtree(PACKAGE_DIR / "nnunet_results", ignore_errors=True)
        shutil.rmtree(PACKAGE_DIR / ".cache", ignore_errors=True)
        logger.info("Model cache cleared.")

    os.makedirs(out_dir, exist_ok=True)

    # Copy inputs into a temporary folder using nnU-Net naming conventions
    des_folder = os.path.join(out_dir, "renamed_image")
    prepare_data_folder(des_folder)
    try:
        rename_dict, rename_back_dict = rename_and_copy_files(in_dir, des_folder)
        datalist_file = os.path.join(des_folder, "renaming.json")
        with open(datalist_file, "w", encoding="utf-8") as fp:
            json.dump(rename_dict, fp, ensure_ascii=False, indent=4)
        logger.info(f"Renaming dict is saved to {datalist_file}")

        model_folder = os.path.join(
            PACKAGE_DIR,
            "nnunet_results",
            f"Dataset{d}_Task{d}_DLMUSEV2",
            f"{tr}__{p}__{c}",
        )
        logger.info(f"Using model folder: {model_folder}")
        if not os.path.exists(model_folder):
            logger.info("DLMUSE model not found, downloading...")
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id="nichart/DLMUSE", local_dir=PACKAGE_DIR)
            logger.info("DLMUSE model has been successfully downloaded!")
        else:
            logger.info("Loading the model...")

        torch_device = _setup_device(device)

        # nnU-Net requires these to be set at import time; the model folder is
        # passed explicitly, so their values are not otherwise used.
        os.environ["nnUNet_raw"] = "/nnunet_raw/"
        os.environ["nnUNet_preprocessed"] = "/nnunet_preprocessed"
        os.environ["nnUNet_results"] = "/nnunet_results"

        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor_cls = nnUNetPredictor
        if low_memory:
            from DLMUSE.predictor import LowMemNNUNetPredictor

            predictor_cls = LowMemNNUNetPredictor
            logger.info("Low memory mode enabled.")

        # Keep the outputs consistent
        torch.use_deterministic_algorithms(True)

        predictor = predictor_cls(
            tile_step_size=step_size,
            use_gaussian=True,
            use_mirroring=not disable_tta,
            perform_everything_on_device=True,
            device=torch_device,
            verbose=verbose,
            verbose_preprocessing=verbose,
            allow_tqdm=not disable_progress_bar,
        )
        predictor.initialize_from_trained_model_folder(
            model_folder, folds, checkpoint_name=chk
        )
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

        # Convert the output names back to the original names
        for filename in os.listdir(out_dir):
            original_name = rename_back_dict.get(filename)
            if original_name is None:
                continue
            os.rename(
                os.path.join(out_dir, filename),
                os.path.join(out_dir, original_name),
            )
    finally:
        shutil.rmtree(des_folder, ignore_errors=True)

    logger.info("Inference Process Done!")

    if qc:
        from DLMUSE.qc import QC_CSV_NAME, run_qc

        # With -num_parts > 1 only this part's outputs exist
        masks = [
            os.path.join(out_dir, name)
            for name in rename_back_dict.values()
            if os.path.isfile(os.path.join(out_dir, name))
        ]
        # QC output goes to <out_dir>/DLMUSE_QC.csv (file|status|issues), not the console
        if masks:
            qc_csv = QC_CSV_NAME if num_parts == 1 else f"DLMUSE_QC_part{part_id}.csv"
            run_qc(masks, os.path.join(out_dir, qc_csv), thresholds=qc_thresholds)
