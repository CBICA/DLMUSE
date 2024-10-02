"""Setup tool for DLMUSE."""

from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="DLMUSE",
    version="1.0.3",
    description="DLMUSE - Deep Learning MUlti-atlas region Segmentation utilizing Ensembles of registration algorithms and parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Guray Erus, Vishnu Bashyam, George Aidinis, Kyunglok Baik, Alexander Getka, Wu Di",
    author_email="software@cbica.upenn.edu",
    maintainer="George Aidinis, Spiros Maggioros, Kyunglok Baik, Alexander Getka",
    maintainer_email="aidinisg@pennmedicine.upenn.edu, Spiros.Maggioros@pennmedicine.upenn.edu, kyunglok.baik@pennmedicine.upenn.edu, alexander.getka@pennmedicine.upenn.edu",
    download_url="https://github.com/CBICA/DLMUSE/",
    url="https://github.com/CBICA/DLMUSE/",
    packages=find_packages(exclude=["tests", ".github"]),
    python_requires=">=3.8",
    install_requires=required,
    entry_points={"console_scripts": ["DLMUSE = DLMUSE.__main__:main"]},
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    license="By installing/using DLMUSE, the user agrees to the following license: See https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html",
    keywords=[
        "deep learning",
        "image segmentation",
        "semantic segmentation",
        "medical image analysis",
        "medical image segmentation",
        "nnU-Net",
        "nnunet",
    ],
    package_data={
        "DLMUSE": ["VERSION"]
    },
)
