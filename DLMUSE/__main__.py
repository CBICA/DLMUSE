import argparse
import logging
from importlib.metadata import version

from DLMUSE.dlmuse_pipeline import run_dlmuse_pipeline
from DLMUSE.qc import (
    MAX_ASYMMETRY_INDEX,
    MAX_WRONG_SIDE_FRACTION,
    MIDLINE_TOLERANCE_MM,
    MIN_PAIRS_FOR_PLANE_FIT,
)
from DLMUSE.utils import set_random_seed

VERSION = version("DLMUSE")


def main() -> None:
    prog = "DLMUSE"
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
            [--low_memory]   Reduce host RAM usage during inference.
            [--qc]           Run L/R hemisphere ROI integrity QC on the outputs (writes DLMUSE_QC.csv: file|status|issues).
              [--qc_thresholds] JSON dict of only the thresholds to override, e.g. '{{"max_wrong_side": 0.2}}'. Keys:
                  max_wrong_side    Max fraction of an ROI past the midline before WARN (default: {WS})
                  midline_tolerance Distance (mm) past the midline that is ignored (default: {TOL})
                  max_asymmetry     Max L/R volume asymmetry index before WARN (default: {ASYM})
                  min_pairs         Min bilateral ROI pairs needed to fit the midline plane (default: {MINP})
                  QC_detail         1 = also write the per-ROI DLMUSE_QC_detail.csv (default: 0)
            [-h, --help]    Show this help message and exit.
            [-V, --version] Show program's version number and exit.
        EXAMPLE USAGE:
            DLMUSE -i       /path/to/input     \\
                   -o       /path/to/output    \\
                   -device  cpu|cuda|mps

        """.format(
            VERSION=VERSION,
            WS=MAX_WRONG_SIDE_FRACTION,
            TOL=MIDLINE_TOLERANCE_MM,
            ASYM=MAX_ASYMMETRY_INDEX,
            MINP=MIN_PAIRS_FOR_PLANE_FIT,
        ),
        add_help=False,
    )

    # Required Arguments
    parser.add_argument(
        "-i",
        "--in_dir",
        type=str,
        required=True,
        help="[REQUIRED] Input folder with LPS oriented T1 sMRI Intra Cranial Volumes (ICV) in Nifti format (nii.gz).",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
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
        "-h", "--help", action="help", help="Show this help message and exit."
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"{prog}: v{VERSION}.",
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
        help="Set this flag to disable progress bar. Recommended for HPC environments (non interactive "
        "jobs)",
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="Set this flag to clear any cached models before running. This is recommended if a previous download failed.",
    )
    parser.add_argument(
        "--low_memory",
        action="store_true",
        help="Reduce host RAM usage: logits are kept on the device until they are reduced to a label map, "
        "and segmentations are exported in the main process (no export worker pool). "
        "Ignored when --save_probabilities is set.",
    )
    parser.add_argument(
        "--qc",
        action="store_true",
        help="Run a quality-control step on the output segmentations that checks left/right hemisphere "
        "ROI integrity (each lateralized MUSE ROI lies in its own hemisphere and has a contralateral "
        "counterpart). One row per file (file, status, issues) is saved to <out_dir>/DLMUSE_QC.csv "
        "(nothing is printed to the console).",
    )
    parser.add_argument(
        "--qc_thresholds",
        type=str,
        default=None,
        help="[QC] JSON dict with only the QC thresholds to override, e.g. "
        '\'{"max_wrong_side": 0.2, "midline_tolerance": 3}\'. Keys: '
        f"max_wrong_side (max fraction 0-1 of an ROI past the midline before WARN, default {MAX_WRONG_SIDE_FRACTION}); "
        f"midline_tolerance (mm past the midline that is ignored, default {MIDLINE_TOLERANCE_MM}); "
        f"max_asymmetry (max L/R volume asymmetry |L-R|/((L+R)/2) before WARN, default {MAX_ASYMMETRY_INDEX}); "
        f"min_pairs (min bilateral ROI pairs to fit the midline plane, default {MIN_PAIRS_FOR_PLANE_FIT}); "
        "QC_detail (1 = also write the per-ROI <out_dir>/DLMUSE_QC_detail.csv, default 0).",
    )
    parser.add_argument(
        "--disable_tta",
        action="store_true",
        help="[nnUnet Arg] Set this flag to disable test time data augmentation in the form of mirroring. "
        "Faster, but less accurate inference. Not recommended.",
    )
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
        "out-of-RAM issues. Not used with --low_memory. Default: 2",
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

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Set random seed to a fixed value
    set_random_seed(42)

    run_dlmuse_pipeline(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        device=args.device,
        verbose=args.verbose,
        save_probabilities=args.save_probabilities,
        continue_prediction=args.continue_prediction,
        disable_progress_bar=args.disable_progress_bar,
        clear_cache=args.clear_cache,
        disable_tta=args.disable_tta,
        d=args.d,
        p=args.p,
        tr=args.tr,
        c=args.c,
        f=[args.f],
        step_size=args.step_size,
        chk=args.chk,
        npp=args.npp,
        nps=args.nps,
        prev_stage_predictions=args.prev_stage_predictions,
        num_parts=args.num_parts,
        part_id=args.part_id,
        low_memory=args.low_memory,
        qc=args.qc,
        qc_thresholds=args.qc_thresholds,
    )


if __name__ == "__main__":
    main()
