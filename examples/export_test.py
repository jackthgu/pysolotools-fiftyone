import pdb
import argparse
import json
import os
import os.path
import pathlib
import sys

import fiftyone
import fiftyone.utils.data
import Imath
import numpy as np
import numpy.random
import OpenEXR as exr
import PIL.Image
from pysolotools_fiftyone.exr_utils import convert_exr_to_png
from pyquaternion import Quaternion
from pysolotools.consumers.solo import Solo
from pysolotools.core.models import (
    BoundingBox2DAnnotation,
    BoundingBox3DAnnotation,
    DepthAnnotation,
    InstanceSegmentationAnnotation,
    KeypointAnnotation,
    NormalAnnotation,
    PixelPositionAnnotation,
    RGBCameraCapture,
    SemanticSegmentationAnnotation,
)

from pysolotools_fiftyone.bounding_box_3d import BBox3D

from pysolotools_fiftyone.solo_fiftyone import SoloDatasetImporter
import pysolotools_fiftyone.exr_utils




name = "solo_dataset"
if fiftyone.dataset_exists(name):
    fiftyone.delete_dataset(name)

dataset = fiftyone.Dataset()
dataset.name = name
dataset.persistent = True

dataset.add_group_field("group", default="rgb")

importer = SoloDatasetImporter('/media/jack/F21A5BFC1A5BBBF3/mtmc_data/solo/', max_samples=5)
dataset.add_importer(importer)
view = dataset.view()

slices = view.select_group_slices()
slices.draw_labels('./test')


