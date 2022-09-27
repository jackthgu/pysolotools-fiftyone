#!/usr/bin/env python
"""
Installs pysolotools-fiftyone
"""

import io
import json
import os
from os.path import dirname, realpath

from setuptools import setup, find_packages

# Package meta-data.
NAME = "pysolotools-fiftyone"
DESCRIPTION = "Voxel fiftyone integration for SOLO"
URL = "https://https://github.com/Unity-Technologies/pysolotools-fiftyone"
EMAIL = "computer-vision@unity3d.com"
AUTHOR = "Unity Technologies"
REQUIRES_PYTHON = ">=3.8"
FALL_BACK_VERSION = "0.3.16"


here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

try:
    with io.open(
        os.path.join(here, "github_release_version.json"), encoding="utf-8"
    ) as f:
        VERSION = json.loads(f.read()).get("version", FALL_BACK_VERSION)
except FileNotFoundError:
    VERSION = FALL_BACK_VERSION


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def _read_requirements():
    requirements = f"{dirname(realpath(__file__))}/requirements.txt"
    with open(requirements) as f:
        results = []
        for line in f:
            line = line.strip()
            if "-i" not in line:
                results.append(line)
        return results


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    py_modules=[NAME],
    include_package_data=True,
    license="MIT",
    install_requires=_read_requirements(),
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": ["pysolotools-fiftyone=pysolotools_fiftyone.solo_fiftyone:cli"]
    },
)