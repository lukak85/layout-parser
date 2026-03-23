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
import torch
from PIL import Image
import numpy as np

from .nemotron_page_elements_v3.model import define_model
from .nemotron_page_elements_v3.utils import plot_sample, postprocess_preds_page_element, reformat_for_plotting

from .catalog import MODEL_CATALOG, LABEL_MAP_CATALOG
from ..base_layoutmodel import BaseLayoutModel
from ...elements import Rectangle, TextBlock, Layout

__all__ = ["NemotronLayoutModel"]


class NemotronLayoutModel(BaseLayoutModel):

    DEPENDENCIES = ["nemotron"]
    DETECTOR_NAME = "nemotron"
    MODEL_CATALOG = MODEL_CATALOG

    def __init__(
        self,
        label_map=None,
    ):
        if label_map is None:
            label_map = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}

        self.label_map = label_map
        self._create_model()

    def _create_model(self):
        self.model = define_model("page_element_v3")

    def gather_output(self, boxes, labels, scores, image_shape):
        """
        xywhn = boxes.xywhn.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        """

        layout = Layout()

        for score, box, label in zip(scores, boxes, labels):
            if score < 0.1:
                continue

            x_1, y_1, x_2, y_2 = box
            x_1 *= image_shape[1]
            x_2 *= image_shape[1]
            y_1 *= image_shape[0]
            y_2 *= image_shape[0]

            label = self.label_map.get(label, label)

            cur_block = TextBlock(
                Rectangle(x_1, y_1, x_2, y_2), type=label, score=score
            )
            layout.append(cur_block)

        return layout

    def detect(self, path):
        """Detect the layout of a given image.

        Args:
            image (:obj:`np.ndarray` or `PIL.Image`): The input image to detect.

        Returns:
            :obj:`~layoutparser.Layout`: The detected layout of the input image
        """

        img = self.image_loader(path)
        with torch.inference_mode():
            x = self.model.preprocess(img)
            preds = self.model(x, img.shape)[0]

        boxes, labels, scores = postprocess_preds_page_element(preds, self.model.thresholds_per_class, self.model.labels)

        return self.gather_output(boxes, labels, scores, img.shape)

    def image_loader(self, path):
        img = Image.open(path).convert("RGB")
        return np.array(img)
