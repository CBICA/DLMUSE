import os
import shutil
from typing import Tuple
import random

import numpy as np
import torch


def prepare_data_folder(folder_path: str) -> None:
    """
    prepare data folder, create one if not exist
    if exist, empty the folder
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def rename_and_copy_files(src_folder: str, des_folder: str) -> Tuple[dict, dict]:
    """
    Input:
         src_folder: a user input folder, name could be anything, will be convert into nnUnet
         format internally

         des_folder: where you want to store your folder

    Returns:
         rename_dict : a dictionary mapping your original name into nnUnet format name
         rename_back_dict:  a dictionary will be use to mapping backto the original name

    """
    if not os.path.exists(src_folder):
        raise FileNotFoundError(f"Source folder '{src_folder}' does not exist.")
    if not os.path.exists(des_folder):
        raise FileNotFoundError(f"Source folder '{des_folder}' does not exist.")

    files = os.listdir(src_folder)
    rename_dict = {}
    rename_back_dict = {}

    for idx, filename in enumerate(files):
        old_name = os.path.join(src_folder, filename)
        if not os.path.isfile(old_name):  # We only want files!
            continue
        rename_file = f"case_{idx: 04d}_0000.nii.gz"
        rename_back = f"case_{idx: 04d}.nii.gz"
        new_name = os.path.join(des_folder, rename_file)
        try:
            shutil.copy2(old_name, new_name)
            rename_dict[filename] = rename_file
            rename_back_dict[rename_back] = "DLMUSE_mask_" + filename
        except Exception as e:
            print(f"Error copying file '{filename}' to '{new_name}': {e}")

    return rename_dict, rename_back_dict

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False