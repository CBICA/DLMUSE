![macos tests](https://github.com/CBICA/DLMUSE/actions/workflows/macos-build.yml/badge.svg)
![ubuntu tests](https://github.com/CBICA/DLMUSE/actions/workflows/ubuntu-build.yml/badge.svg)
![PyPI Stable](https://img.shields.io/pypi/v/DLMUSE)

# DLMUSE - Deep Learning MUlti-atlas region Segmentation utilizing Ensembles of registration algorithms and parameters

## Overview

DLMUSE uses a trained [nnUNet](https://github.com/MIC-DKFZ/nnUNet) model to compute the segmentation of the brain into [MUSE](https://www.med.upenn.edu/cbica/sbia/muse.html) ROIs from the nifti image of the Intra Cranial Volume (ICV - see [DLICV method](https://github.com/CBICA/DLICV)), oriented in _**LPS**_ orientation. It produces a MUSE label map for each input image, and can optionally run a quality-control (QC) check of left/right hemisphere ROI integrity on the results.

Please make sure to use post-skull stripped T1 brain images (.nii.gz) as inputs or use [NiChart_DLMUSE](https://github.com/CBICA/NiChart_DLMUSE) for the full processing pipeline from RAW T1.

## Installation

### As a python package

```bash
pip install DLMUSE
```

### Directly from this repository

```bash
git clone https://github.com/CBICA/DLMUSE
cd DLMUSE
pip install -e .
```

### Installing PyTorch
Depending on your system configuration and supported CUDA version, you may need to follow the [PyTorch Installation Instructions](https://pytorch.org/get-started/locally/).

## Usage

A pre-trained nnUNet model can be found at our [hugging face account](https://huggingface.co/nichart/DLMUSE/tree/main). Feel free to use it under the package's [license](LICENSE).

### From command line
```bash
DLMUSE -i "input_folder" -o "output_folder" -device cpu
```
### In-code usage
```python
from DLMUSE import run_dlmuse_pipeline
...
run_dlmuse_pipeline(in_dir, out_dir, device)
```

For more details, please refer to

```bash
DLMUSE -h
```

### Inputs and outputs
- **Input:** a folder of skull-stripped, LPS-oriented `.nii.gz` images. Files with other extensions are skipped with a warning.
- **Output:** one segmentation per input, `DLMUSE_mask_<input name>.nii.gz`, written to the output folder. Labels use the consecutive MUSE indices listed in `DLMUSE/shared/dicts/MUSE_mapping_consecutive_indices.csv`.

The random seed is fixed and deterministic algorithms are enabled, so repeated runs on the same machine give the same results.

### Reducing memory usage
DLMUSE predicts 152 classes, so nnU-Net's default export path can use a lot of system RAM. If inference runs out of memory, add the `--low_memory` flag:

```bash
DLMUSE -i "input_folder" -o "output_folder" -device cuda --low_memory
```

With this flag, the model's raw outputs stay on the device until they are reduced to a label map, and results are written in the main process instead of in export worker processes. It has no effect when `--save_probabilities` is set. The in-code equivalent is `run_dlmuse_pipeline(in_dir, out_dir, device, low_memory=True)`.

### Quality control (QC)
Add `--qc` to check the left/right hemisphere integrity of each output segmentation:

```bash
DLMUSE -i "input_folder" -o "output_folder" -device cuda --qc
```

For every lateralized MUSE ROI, QC checks that the ROI lies in its own hemisphere and has a counterpart on the other side. The midline is fitted through the midpoints of the left/right ROI centroids, so it follows head tilt and rotation. Each ROI pair is rated:

| Status | Meaning |
|---|---|
| `FAIL` | A left or right ROI is missing, or an ROI's centroid lies in the wrong hemisphere. |
| `WARN` | Too much of an ROI crosses the midline (`max_wrong_side`), or the left/right volumes differ too much (`max_asymmetry`). |
| `PASS` | No problems found. |

Results are written to `<out_dir>/DLMUSE_QC.csv` with one row per file. Nothing is printed to the console.

| Column | Content |
|---|---|
| `file` | Segmentation file name |
| `status` | Worst status across the file's ROI pairs |
| `issues` | Every issue in the file, separated by `; `. File-level notes come first (e.g. labels not in the MUSE mapping, a tilted midline, empty or unreadable segmentation), then each ROI issue as `<roi> (L:<left index>/R:<right index>): <issue>`, e.g. `inf_lat_vent (L:20/R:19): high L/R volume asymmetry (1.12)` |

Some ROIs (e.g. `vessel`) are never predicted. These are noted as `absent in both hemispheres` and do not affect the status. File-level notes also leave the status unchanged, except for an empty or unreadable segmentation, which is `FAIL`.

#### QC thresholds and detailed output
Pass `--qc_thresholds` a JSON dict that contains only the values you want to change. Keys you leave out keep their defaults:

```bash
DLMUSE -i "input_folder" -o "output_folder" --qc --qc_thresholds '{"max_wrong_side": 0.2, "QC_detail": 1}'
```

| Key | Default | Meaning |
|---|---|---|
| `max_wrong_side` | `0.1` | `WARN` when more than this fraction (0-1) of an ROI lies in the opposite hemisphere beyond `midline_tolerance` |
| `midline_tolerance` | `2.0` | Distance (mm) past the midline that is not counted as crossing it |
| `max_asymmetry` | `1.0` | `WARN` when the L/R volume asymmetry index \|L-R\|/((L+R)/2) exceeds this value (1.0 = one side is 3x the other) |
| `min_pairs` | `10` | Minimum number of left/right ROI pairs needed to fit the midline; with fewer, the plane x = mean(x) is used |
| `QC_detail` | `0` | `1` also writes `<out_dir>/DLMUSE_QC_detail.csv`, with one row per file and ROI pair (voxel counts, volumes, asymmetry index, centroid distances to the midline, wrong-side fractions, status and issues) |

Unknown keys and invalid values are rejected before inference starts. With `-num_parts > 1`, each part writes `DLMUSE_QC_part<part_id>.csv` (and `DLMUSE_QC_part<part_id>_detail.csv`).

In code, pass the same settings as a dict or a JSON string:

```python
run_dlmuse_pipeline(in_dir, out_dir, device, qc=True, qc_thresholds={"max_asymmetry": 1.5, "QC_detail": 1})
```

## \[Windows Users\] Troubleshooting model download failures
Our model download process creates several deep directory structures. If you are on Windows and your model download process fails, it may be due to Windows file path limitations.

To enable long path support in Windows 10, version 1607, and later, the registry key `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem LongPathsEnabled (Type: REG_DWORD)` must exist and be set to 1.

If this affects you, we recommend re-running DLMUSE with the `--clear_cache` flag set on the first run.

## Citation

Bashyam VM, Erus G, Cui Y, Wu D, Hwang G, Getka A, Singh A, Aidinis G, Baik K, Melhem R, Mamourian E, Doshi J, Davison A, Nasrallah IM, Davatzikos C; Alzheimer’s Disease Neuroimaging Initiative; iSTAGING Consortium. DLMUSE: Robust Brain Segmentation in Seconds Using Deep Learning. Radiol Artif Intell. 2025 Nov;7(6):e240299. doi: 10.1148/ryai.240299. PMID: 40960397; PMCID: PMC12661376.

## Contact

For more information, please contact [CBICA Software](mailto:software@cbica.upenn.edu).

## For Developers

Contributions are welcome! Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for more information on how to report bugs, suggest enhancements, and contribute code.
Please make sure to write tests for new code and run them before submitting a pull request.
