"""
Memory-light nnU-Net predictor for DLMUSE.

The default nnU-Net export path moves the full (num_classes, X, Y, Z) logits
array to host RAM, pickles a copy into each export worker, resamples it in
float64 and applies softmax in float32 before taking the argmax. With 152
MUSE classes this dominates host memory usage.

LowMemNNUNetPredictor keeps the logits on the device until they are reduced
to a label map and exports in the main process (no worker pool). When
resampling back to the original spacing is required, channels are resampled
in small chunks with nnU-Net's own resampling function while a running
argmax is maintained, so only a few channels are in host RAM at once.
"""

import os
from typing import Any, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import empty_cache
from torch._dynamo import OptimizedModule


class LowMemNNUNetPredictor(nnUNetPredictor):
    # number of channels resampled at once when resampling is needed
    resample_chunk_size: int = 8

    def predict_from_data_iterator(
        self,
        data_iterator: Any,
        save_probabilities: bool = False,
        num_processes_segmentation_export: int = 1,
    ) -> List[Optional[np.ndarray]]:
        if save_probabilities:
            # probabilities require the full array anyway -> default nnU-Net path
            ret_default: List[
                Optional[np.ndarray]
            ] = super().predict_from_data_iterator(
                data_iterator, save_probabilities, num_processes_segmentation_export
            )
            return ret_default

        ret: List[Optional[np.ndarray]] = []
        for preprocessed in data_iterator:
            data = preprocessed["data"]
            if isinstance(data, str):
                delfile = data
                data = torch.from_numpy(np.load(data))
                os.remove(delfile)

            ofile = preprocessed["ofile"]
            properties = preprocessed["data_properties"]
            name = os.path.basename(ofile) if ofile is not None else str(data.shape)
            print("\nPredicting %s (low memory mode)" % name)

            logits = self._predict_logits_keep_on_device(data)
            del data
            segmentation = self._logits_to_segmentation(logits, properties)
            del logits
            empty_cache(self.device)

            if ofile is not None:
                rw = self.plans_manager.image_reader_writer_class()
                rw.write_seg(
                    segmentation, ofile + self.dataset_json["file_ending"], properties
                )
                print(f"done with {os.path.basename(ofile)}")
                ret.append(None)
            else:
                ret.append(segmentation)

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        compute_gaussian.cache_clear()
        empty_cache(self.device)
        return ret

    def _predict_logits_keep_on_device(self, data: torch.Tensor) -> torch.Tensor:
        """Like predict_logits_from_preprocessed_data, but for a single fold the
        logits are not moved to the CPU. (If the device runs out of memory,
        nnU-Net itself falls back to CPU results arrays.)"""
        if len(self.list_of_parameters) != 1:
            return self.predict_logits_from_preprocessed_data(data)

        n_threads = torch.get_num_threads()
        torch.set_num_threads(min(default_num_processes, n_threads))
        try:
            params = self.list_of_parameters[0]
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)
            return self.predict_sliding_window_return_logits(data)
        finally:
            torch.set_num_threads(n_threads)

    @torch.no_grad()  # type: ignore[misc]
    def _logits_to_segmentation(
        self, logits: torch.Tensor, properties: dict
    ) -> np.ndarray:
        """Mirrors convert_predicted_logits_to_segmentation_with_correct_shape
        (nnunetv2/inference/export_prediction.py) for the argmax case."""
        lm = self.label_manager
        assert (
            not lm.has_regions
        ), "LowMemNNUNetPredictor does not support region-based models"
        seg_dtype = np.uint8 if len(lm.foreground_labels) < 255 else np.uint16
        torch_seg_dtype = torch.uint8 if seg_dtype == np.uint8 else torch.int16

        target_shape = [
            int(i) for i in properties["shape_after_cropping_and_before_resampling"]
        ]

        if list(logits.shape[1:]) == target_shape:
            # no resampling needed: argmax directly on the device.
            # argmax(softmax(x)) == argmax(x), softmax is monotonic.
            segmentation = (
                logits.argmax(0).to(torch_seg_dtype).cpu().numpy().astype(seg_dtype)
            )
        else:
            spacing_transposed = [
                properties["spacing"][i] for i in self.plans_manager.transpose_forward
            ]
            current_spacing = (
                self.configuration_manager.spacing
                if len(self.configuration_manager.spacing) == len(target_shape)
                else [spacing_transposed[0], *self.configuration_manager.spacing]
            )
            new_spacing = spacing_transposed

            best_val: Optional[np.ndarray] = None
            best_idx = np.zeros(target_shape, dtype=seg_dtype)
            n_channels = logits.shape[0]
            for start in range(0, n_channels, self.resample_chunk_size):
                stop = min(start + self.resample_chunk_size, n_channels)
                chunk = logits[start:stop].cpu()
                in_dtype = chunk.numpy().dtype
                resampled = self.configuration_manager.resampling_fn_probabilities(
                    chunk, target_shape, current_spacing, new_spacing
                )
                del chunk
                if isinstance(resampled, torch.Tensor):
                    resampled = resampled.numpy()
                # nnU-Net returns the resampled array in the input dtype (fp16)
                resampled = resampled.astype(in_dtype, copy=False)
                for c in range(resampled.shape[0]):
                    if best_val is None:
                        best_val = resampled[c].copy()
                        best_idx[:] = start + c
                    else:
                        mask = resampled[c] > best_val  # strict: ties keep first class
                        best_val[mask] = resampled[c][mask]
                        best_idx[mask] = start + c
                del resampled
            del best_val
            segmentation = best_idx

        # put segmentation in bbox (revert cropping)
        seg_reverted = np.zeros(properties["shape_before_cropping"], dtype=seg_dtype)
        slicer = bounding_box_to_slice(properties["bbox_used_for_cropping"])
        seg_reverted[slicer] = segmentation
        del segmentation

        # revert transpose
        return seg_reverted.transpose(self.plans_manager.transpose_backward)
