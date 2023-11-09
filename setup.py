"""Setup tool for DLMUSE."""

import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely."""

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="DLMUSE",
    version=read("DLMUSE", "VERSION"),
    description="DLMUSE - Deep Learning MUlti-atlas region Segmentation utilizing Ensembles of registration algorithms and parameters",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Ashish Singh, Guray Erus, Vishnu Bashyam, George Aidinis",
    author_email="software@cbica.upenn.edu",
    maintainer="George Aidinis",
    maintainer_email="aidinisg@pennmedicine.upenn.edu",
    download_url="https://github.com/CBICA/DLMUSE/",
    url="https://github.com/CBICA/DLMUSE/",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={"console_scripts": ["DLMUSE = DLMUSE.__main__:main"]},
    extras_require={"test": read_requirements("requirements-test.txt")},
    classifiers=[
                    "Intended Audience :: Developers",
                    "Intended Audience :: Science/Research",
                    "Intended Audience :: Healthcare Industry",
                    "Programming Language :: Python :: 3",
                    "Topic :: Scientific/Engineering :: Artificial Intelligence",
                    "Topic :: Scientific/Engineering :: Image Processing",
                    "Topic :: Scientific/Engineering :: Medical Science Apps.",
                ],
    license="By installing/using DeepMRSeg, the user agrees to the following license: See https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html",
    keywords = [
                    'deep learning',
                    'image segmentation',
                    'semantic segmentation',
                    'medical image analysis',
                    'medical image segmentation',
                    'nnU-Net',
                    'nnunet'
                ], 
    package_data={"DLMUSE": ["VERSION","dicts/*"]},
)