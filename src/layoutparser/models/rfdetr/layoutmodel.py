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
from rfdetr_doclayout.rfdetr import RfDetrDoclayout

from .catalog import MODEL_CATALOG, LABEL_MAP_CATALOG
from ..base_layoutmodel import BaseLayoutModel
from ...elements import Rectangle, TextBlock, Layout

__all__ = ["RFDETRLayoutModel"]


class RFDETRLayoutModel(BaseLayoutModel):

    DEPENDENCIES = ["rfdetr"]
    DETECTOR_NAME = "rfdetr"
    MODEL_CATALOG = MODEL_CATALOG

    def __init__(
        self,
        label_map=None
    ):
        self.model = RfDetrDoclayout()

        if label_map is None:
            label_map = LABEL_MAP_CATALOG["DocLayNet"]
        else:
            label_map = LABEL_MAP_CATALOG[label_map]

        self.label_map = label_map

    def gather_output(self, labels, boxes, image_id=1):
        json_results = []

        for box, label in zip(boxes, labels):
            x = box[0]
            y = box[1]
            w = box[2] - box[0]
            h = box[3] - box[1]

            json_results.append(
                {
                    "image_id": image_id,
                    "label": label,
                    "bbox": [x, y, w, h],
                    "score": float(1),
                }
            )

        layout_for_lp = Layout()

        # for score, box, label in zip(scores, boxes, labels):
        for res in json_results:
            score = res["score"]
            label = self.label_map.get(res["label"], res["label"])
            box = res["bbox"]

            x_1, y_1, w, h = box
            x_2 = x_1 + w
            y_2 = y_1 + h

            cur_block = TextBlock(
                Rectangle(x_1, y_1, x_2, y_2), type=label, score=score
            )
            layout_for_lp.append(cur_block)

        return layout_for_lp

    def detect(self, image):
        """Detect the layout of a given image.

        Args:
            image (:obj:`np.ndarray` or `PIL.Image`): The input image to detect.

        Returns:
            :obj:`~layoutparser.Layout`: The detected layout of the input image
        """

        _, labels, boxes, _ = self.model.predict(image.as_posix())
        layout = self.gather_output(labels, boxes)

        return layout

    def image_loader(self, image):
        return image
