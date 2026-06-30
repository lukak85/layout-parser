# Copyright 2021 The Layout Parser team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import PIL
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

from .catalog import MODEL_CATALOG
from ..base_layoutmodel import BaseLayoutModel
from ...elements import Rectangle, TextBlock, Layout

__all__ = ["RecursiveXYCutLayoutModel"]


class RecursiveXYCutLayoutModel(BaseLayoutModel):

    DEPENDENCIES = []
    DETECTOR_NAME = "recursive_xycut"
    MODEL_CATALOG = MODEL_CATALOG

    def __init__(
            self,
            n=10,
            ignoreBottomTop=True,
            axis=0
    ):
        self.n = n
        self.ignoreBottomTop = ignoreBottomTop
        self.axis = axis

    def detect(self, image: str) -> Layout:
        image = self.image_loader(image).convert('L')
        image_arr = np.asarray(image)
        # distance for peaks
        distance = image_arr.shape[0 if self.axis == 1 else 1] / self.n
        # Sum the pixels along given axis
        sum_vals = image_arr.sum(axis=self.axis)
        # Get the indices of the peaks
        peaks, _ = find_peaks(sum_vals, distance=distance)
        # Temp variable to create segment lines i.e. 0 out the required values.
        temp = np.ones(image_arr.shape)
        # Skip top and bottom segmentation or not (depends on the param)
        # for peak in peaks[1:-1 if ignoreBottomTop else ]:
        for peak in peaks[1:-1] if self.ignoreBottomTop else peaks:
            if self.axis == 1:
                temp[range(peak - 2, peak + 2)] = 0
            else:
                temp[:, range(peak - 2, peak + 2)] = 0
        si = Image.fromarray(np.uint8(image_arr * temp))
        plt.imshow(si)
        plt.axis("off")
        plt.show()
        quit()

    def image_loader(self, image: str) -> "PIL.Image.Image":
        img = Image.open(image)
        return img