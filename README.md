# DLMUSE - Deep Learning MUlti-atlas region Segmentation utilizing Ensembles of registration algorithms and parameters

## Overview

DLMUSE uses a trained [nnUNet](https://github.com/MIC-DKFZ/nnUNet) model to compute the segmentation of the brain into [MUSE](https://www.med.upenn.edu/cbica/sbia/muse.html) ROIs from the nifti image of the Intra Cranial Volume (ICV - see [DLICV method](https://github.com/CBICA/DLICV)), oriented in _**LPS**_ orientation. It produces the segmented brain, along with a .csv file of the calculated volumes of each ROI.

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
For more details, please refer to

```bash
DLMUSE -h
```

## \[Windows Users\] Troubleshooting model download failures
Our model download process creates several deep directory structures. If you are on Windows and your model download process fails, it may be due to Windows file path limitations. 

To enable long path support in Windows 10, version 1607, and later, the registry key `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem LongPathsEnabled (Type: REG_DWORD)` must exist and be set to 1.

If this affects you, we recommend re-running DLMUSE with the `--clear_cache` flag set on the first run.

## Contact

For more information, please contact [CBICA Software](mailto:software@cbica.upenn.edu).

## For Developers

Contributions are welcome! Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for more information on how to report bugs, suggest enhancements, and contribute code.
Please make sure to write tests for new code and run them before submitting a pull request.
