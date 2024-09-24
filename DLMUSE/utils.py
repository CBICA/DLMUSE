import os
import shutil
from typing import Tuple


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
    files = os.listdir(src_folder)
    rename_dict = {}
    rename_back_dict = {}

    for idx, filename in enumerate(files):
        old_name = os.path.join(src_folder, filename)
        rename_file = f"case_{idx: 04d}_0000.nii.gz"
        rename_back = f"case_{idx: 04d}.nii.gz"
        new_name = os.path.join(des_folder, rename_file)
        shutil.copy2(old_name, new_name)
        rename_dict[filename] = rename_file
        rename_back_dict[rename_back] = "DLMUSE_mask_" + filename

    return rename_dict, rename_back_dict
