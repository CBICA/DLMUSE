def compute_segmentation(input_path, output_path, model_path, **kwargs):
    """
    Compute the MUSE ROIs from the input image using the nnUNet model.
    :param input_path: Path to the input image (single image or directory of images).
    :param output_path: Path for the output image (single image or directory of images).
    :param model_path: Path to the model to be applied, or 'default'.
    :param kwargs: Additional keyword arguments for the predict_from_folder function.
    :return: The output image with the computed volume or a list of output images.
    """
    from nnunet.inference.predict import predict_from_folder

    # FROM nnUNet docs: (https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/nnunet/inference/predict.py#L614)
    # predict_from_folder(  model: str, "model output folder. Will automatically discover the folds "
    #                       input_folder: str, "Must contain all modalities for each patient in the correct order (same as training). Files must be named CASENAME_XXXX.nii.gz where XXXX is the modality identifier (0000, 0001, etc)", required=True
    #                       output_folder: str, "folder for saving predictions"
    #                       folds: Union[Tuple[int], List[int]],"folds to use for prediction. Default is None which means that folds will be detected automatically in the model output folder"
    #                       save_npz: bool, "use this if you want to ensemble these predictions with those of other models. Softmax probabilities will be saved as compresed numpy arrays in output_folder and can be merged between output_folders with merge_predictions.py"
    #                       num_threads_preprocessing: int, "Determines many background processes will be used for data preprocessing. Reduce this if you run into out of memory (RAM) problems. Default: 6"
    #                       num_threads_nifti_save: int, "Determines many background processes will be used for segmentation export. Reduce this if you run into out of memory (RAM) problems. Default: 6"
    #                       lowres_segmentations: Union[str, None], "if model is the highres stage of the cascade then you need to use -l to specify where the segmentations of the corresponding lowres unet are. Here they are required to do a prediction"
    #                       part_id: int, "Used to parallelize the prediction of the folder over several GPUs. If you want to use n GPUs to predict this folder you need to run this command n times with --part_id=0, ... n-1 and --num_parts=n (each with a different GPU (for example via CUDA_VISIBLE_DEVICES=X)"
    #                       num_parts: int, "Used to parallelize the prediction of the folder over several GPUs. If you want to use n GPUs to predict this folder you need to run this command n times with --part_id=0, ... n-1 and --num_parts=n (each with a different GPU (for example via CUDA_VISIBLE_DEVICES=X)"
    #                       tta: bool, "Set to 0 to disable test time data augmentation (speedup of factor 4(2D)/8(3D)), lower quality segmentations"
    #                       mixed_precision: bool = True, 'Predictions are done with mixed precision by default. This improves speed and reduces the required vram. If you want to disable mixed precision you can set this flag. Note that this is not recommended (mixed precision is ~2x faster!)'
    #                       overwrite_existing: bool = True, "Set this to 0 if you need to resume a previous prediction. Default: 1 (=existing segmentations in output_folder will be overwritten)"
    #                       mode: str = 'normal', "'normal' or 'fast' or 'fastest'"
    #                       overwrite_all_in_gpu: bool = None, "can be None, False or True"
    #                       step_size: float = 0.5, "don't touch"
    #                       checkpoint_name: str = "model_final_checkpoint",
    #                       segmentation_export_kwargs: dict = None,
    #                       disable_postprocessing: bool = False):
    
    # FOR THE PRETRAINED MODEL WITH TASK_ID = 802 AKA DLICV
    # Default values for predict_from_folder parameters
    default_params = {
        'folds': [2],
        'save_npz': False,
        'num_threads_preprocessing': 6,
        'num_threads_nifti_save': 6,
        'lowres_segmentations': None,
        'part_id': 0,  # Change this if multiple GPUs are present
        'num_parts': 1,
        'tta': 0,
        'mixed_precision': True,
        'overwrite_existing': False,
        'mode': 'fastest',
        'overwrite_all_in_gpu': 1,
        'step_size': 0.5,
        'checkpoint_name': "model_final_checkpoint",
        'segmentation_export_kwargs': None,
        'disable_postprocessing': False
    }

    # Update default parameters with any additional keyword arguments provided
    params = {**default_params, **kwargs}

    # Call predict_from_folder with updated parameters
    predict_from_folder(model_path,
                        input_path,
                        output_path,
                        **params)
    
    return