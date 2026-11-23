import logging
import os
import random
import shutil
from typing import Dict, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

NIFTI_EXT = ".nii.gz"


def prepare_data_folder(folder_path: str) -> None:
    """Create the folder (and any parents) if it does not already exist."""
    os.makedirs(folder_path, exist_ok=True)


def rename_and_copy_files(
    src_folder: str, des_folder: str
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Copy every .nii.gz image in src_folder into des_folder using nnU-Net naming.

    Returns:
        rename_dict: maps each original filename to its nnU-Net input name
        rename_back_dict: maps each nnU-Net output name to the final output name
    """
    if not os.path.exists(src_folder):
        raise FileNotFoundError(f"Source folder '{src_folder}' does not exist.")
    if not os.path.exists(des_folder):
        raise FileNotFoundError(f"Destination folder '{des_folder}' does not exist.")

    rename_dict: Dict[str, str] = {}
    rename_back_dict: Dict[str, str] = {}

    for idx, filename in enumerate(sorted(os.listdir(src_folder))):
        old_name = os.path.join(src_folder, filename)
        if not os.path.isfile(old_name):
            continue
        if not filename.endswith(NIFTI_EXT):
            logger.warning(f"Skipping '{filename}': not a {NIFTI_EXT} file.")
            continue
        rename_file = f"case_{idx:04d}_0000{NIFTI_EXT}"
        rename_back = f"case_{idx:04d}{NIFTI_EXT}"
        new_name = os.path.join(des_folder, rename_file)
        try:
            shutil.copy2(old_name, new_name)
        except OSError as e:
            logger.error(f"Error copying file '{filename}' to '{new_name}': {e}")
            continue
        rename_dict[filename] = rename_file
        rename_back_dict[rename_back] = "DLMUSE_mask_" + filename

    return rename_dict, rename_back_dict


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
