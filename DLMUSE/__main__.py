import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import warnings

import torch

from .utils import prepare_data_folder, rename_and_copy_files

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

VERSION = 1.0

def main() -> None:
    prog="DLMUSE"
    parser = argparse.ArgumentParser(
        prog=prog,
        description="DLMUSE - MUlti-atlas region Segmentation utilizing Ensembles of registration algorithms and parameters.",
        usage="""
        DLMUSE v{VERSION}
        Segmentation of the brain into MUSE ROIs from the Nifti image (.nii.gz) of the LPS oriented Intra Cranial Volume (ICV - see DLICV method).
        
        Required arguments:
            [-i, --in_dir]   The filepath of the input directory
            [-o, --out_dir]  The filepath of the output directory
        Optional arguments:
            [-device]        cpu|cuda|mps - Depending on your system configuration (default: cuda)
            [-h, --help]    Show this help message and exit.
            [-V, --version] Show program's version number and exit.
        EXAMPLE USAGE:
            DLMUSE -i       /path/to/input     \
                   -o       /path/to/output    \
                   -device  cpu|cuda|mps

        """.format(VERSION=VERSION),
        add_help=False
    )

    # Required Arguments
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        help="[REQUIRED] Input folder with LPS oriented T1 sMRI Intra Cranial Volumes (ICV) in Nifti format (nii.gz).",
    )
    parser.add_argument(
        "-o",
        type=str,
        required=True,
        help="[REQUIRED] Output folder for the segmentation results in Nifti format (nii.gz).",
    )
    
    # Optional Arguments
    parser.add_argument(
        "-device",
        type=str,
        default="cuda",
        required=False,
        help="[Recommended] Use this to set the device the inference should run with. Available options are 'cuda' (GPU), "
        "'cpu' (CPU) or 'mps' (Apple M-series chips supporting 3D CNN).",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=prog + ": v{VERSION}.".format(VERSION=VERSION),
        help="Show the version and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Provides additional details when set.",
    )
    parser.add_argument(
        "--save_probabilities",
        action="store_true",
        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
        "multiple configurations.",
    )
    parser.add_argument(
        "--continue_prediction",
        action="store_true",
        help="Continue an aborted previous prediction (will not overwrite existing files)",
    )
    parser.add_argument(
        "--disable_progress_bar",
        action="store_true",
        required=False,
        default=False,
        help="Set this flag to disable progress bar. Recommended for HPC environments (non interactive "
        "jobs)",
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        required=False,
        default=False,
        help="Set this flag to clear any cached models before running. This is recommended if a previous download failed."
    )
    parser.add_argument(
        "--disable_tta",
        action="store_true",
        required=False,
        default=False,
        help="[nnUnet Arg] Set this flag to disable test time data augmentation in the form of mirroring. "
        "Faster, but less accurate inference. Not recommended.",
    )
    ### DEPRECIATED ####
    # parser.add_argument(
    #     "-m",
    #     type=str,
    #     required=True,
    #     help="Model folder path. The model folder should be named nnunet_results.",
    # )
    parser.add_argument(
        "-d",
        type=str,
        required=False,
        default="903",
        help="[nnUnet Arg] Dataset with which you would like to predict. You can specify either dataset name or id",
    )
    parser.add_argument(
        "-p",
        type=str,
        required=False,
        default="nnUNetPlans",
        help="[nnUnet Arg] Plans identifier. Specify the plans in which the desired configuration is located. "
        "Default: nnUNetPlans",
    )
    parser.add_argument(
        "-tr",
        type=str,
        required=False,
        default="nnUNetTrainer",
        help="[nnUnet Arg] nnU-Net trainer class used for training. "
        "Default: nnUNetTrainer",
    )
    parser.add_argument(
        "-c",
        type=str,
        required=False,
        default="3d_fullres",
        help="[nnUnet Arg] nnU-Net configuration that should be used for prediction. Config must be located "
        "in the plans specified with -p",
    )
    parser.add_argument(
        "-f",
        type=int,
        required=False,
        default=0,
        help="[nnUnet Arg] Specify the folds of the trained model that should be used for prediction. "
        "Default: 0",
    )
    parser.add_argument(
        "-step_size",
        type=float,
        required=False,
        default=0.5,
        help="[nnUnet Arg] Step size for sliding window prediction. The larger it is the faster "
        "but less accurate prediction. Default: 0.5. Cannot be larger than 1. We recommend the default. "
        "Default: 0.5",
    )
    parser.add_argument(
        "-chk",
        type=str,
        required=False,
        default="checkpoint_final.pth",
        help="[nnUnet Arg] Name of the checkpoint you want to use. Default: checkpoint_final.pth",
    )
    parser.add_argument(
        "-npp",
        type=int,
        required=False,
        default=2,
        help="[nnUnet Arg] Number of processes used for preprocessing. More is not always better. Beware of "
        "out-of-RAM issues. Default: 2",
    )
    parser.add_argument(
        "-nps",
        type=int,
        required=False,
        default=2,
        help="[nnUnet Arg] Number of processes used for segmentation export. More is not always better. Beware of "
        "out-of-RAM issues. Default: 2",
    )
    parser.add_argument(
        "-prev_stage_predictions",
        type=str,
        required=False,
        default=None,
        help="[nnUnet Arg] Folder containing the predictions of the previous stage. Required for cascaded models.",
    )
    parser.add_argument(
        "-num_parts",
        type=int,
        required=False,
        default=1,
        help="[nnUnet Arg] Number of separate nnUNetv2_predict call that you will be making. "
        "Default: 1 (= this will predict everything with a single call)",
    )
    parser.add_argument(
        "-part_id",
        type=int,
        required=False,
        default=0,
        help="[nnUnet Arg] If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 "
        "can end with num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set " 
        "-num_parts 5 and use -part_id 0, 1, 2, 3 and 4. Note: You are yourself responsible to make these run on separate GPUs! "
        "Use CUDA_VISIBLE_DEVICES.",
    )
    


    args = parser.parse_args()
    args.f = [args.f]

    if args.clear_cache:
        shutil.rmtree(os.path.join(
            Path(__file__).parent,
            "nnunet_results"
        ))
        shutil.rmtree(os.path.join(
            Path(__file__).parent,
            ".cache"
        ))
        if not args.i or not args.o:
            print("Cache cleared and missing either -i / -o. Exiting.")
            sys.exit(0)

    if not args.i or not args.o:
        parser.error("The following arguments are required: -i, -o")

    # data conversion
    src_folder = args.i  # input folder

    if not os.path.exists(args.o):  # create output folder if it does not exist
        os.makedirs(args.o)

    des_folder = os.path.join(args.o, "renamed_image")

    # check if -i argument is a folder, list (csv), or a single file (nii.gz)
    if os.path.isdir(args.i):  # if args.i is a directory
        src_folder = args.i
        prepare_data_folder(des_folder)
        rename_dic, rename_back_dict = rename_and_copy_files(src_folder, des_folder)
        datalist_file = os.path.join(des_folder, "renaming.json")
        with open(datalist_file, "w", encoding="utf-8") as f:
            json.dump(rename_dic, f, ensure_ascii=False, indent=4)
        print(f"Renaming dic is saved to {datalist_file}")
    else:
        print("Input directory not found. Exiting DLMUSE.")
        sys.exit()

    model_folder = os.path.join(
        Path(__file__).parent,
        "nnunet_results",
        "Dataset%s_Task%s_DLMUSEV2/nnUNetTrainer__nnUNetPlans__%s/"
        % (args.d, args.d, args.c),
    )


    # Check if model exists. If not exist, download using HuggingFace
    print(f"Using model folder: {model_folder}")
    if not os.path.exists(model_folder):
        # HF download model
        print("DLMUSE model not found, downloading...")

        from huggingface_hub import snapshot_download
        local_src = Path(__file__).parent
        snapshot_download(repo_id="nichart/DLMUSE", local_dir=local_src)

        print("DLMUSE model has been successfully downloaded!")
    else:
        print("Loading the model...")

    prepare_data_folder(des_folder)

    assert (
        args.part_id < args.num_parts
    ), "part_id < num_parts. Please see nnUNetv2_predict -h."

    assert args.device in [
        "cpu",
        "cuda",
        "mps",
    ], f"-device must be either cpu, mps or cuda. Got: {args.device}."

    if args.device == "cpu":
        import multiprocessing
        # use half of the available threads in the system.
        torch.set_num_threads(multiprocessing.cpu_count() // 2)
        device = torch.device("cpu")
        print("Running in CPU mode.")
    elif args.device == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
        print("Running in CUDA mode.")
    else:
        device = torch.device("mps")
        print("Running in MPS mode.")

    # exports for nnunetv2 purposes
    os.environ["nnUNet_raw"] = "/nnunet_raw/"
    os.environ["nnUNet_preprocessed"] = "/nnunet_preprocessed"
    os.environ["nnUNet_results"] = (
        "/nnunet_results"  # where model will be located (fetched from HF)
    )

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    # Initialize nnUnetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=args.verbose,
        verbose_preprocessing=args.verbose,
        allow_tqdm=not args.disable_progress_bar,
    )

    # Retrieve the model and it's weight
    predictor.initialize_from_trained_model_folder(
        model_folder, args.f, checkpoint_name=args.chk
    )

    # Final prediction
    predictor.predict_from_files(
        des_folder,
        args.o,
        save_probabilities=args.save_probabilities,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=args.npp,
        num_processes_segmentation_export=args.nps,
        folder_with_segs_from_prev_stage=args.prev_stage_predictions,
        num_parts=args.num_parts,
        part_id=args.part_id,
    )

    # After prediction, convert the image name back to original
    files_folder = args.o

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

    print("Inference Process Done!")


if __name__ == "__main__":
    main()
