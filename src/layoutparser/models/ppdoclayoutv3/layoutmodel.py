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
from paddleocr import LayoutDetection

from .catalog import MODEL_CATALOG, LABEL_MAP_CATALOG
from ..base_layoutmodel import BaseLayoutModel
from ...elements import Rectangle, TextBlock, Layout

__all__ = ["PPDocLayoutV3LayoutModel"]


class PPDocLayoutV3LayoutModel(BaseLayoutModel):

    DEPENDENCIES = ["ppdoclayoutv3"]
    DETECTOR_NAME = "ppdoclayoutv3"
    MODEL_CATALOG = MODEL_CATALOG

    def __init__(
        self,
        model="PP-DocLayoutV3"
    ):
        self.model = LayoutDetection(model_name=model)

    def gather_output(self, pp_result, image_id=1):
        json_results = []

        for box in pp_result["boxes"]:
            x = box["coordinate"][0]
            y = box["coordinate"][1]
            w = box["coordinate"][2] - box["coordinate"][0]
            h = box["coordinate"][3] - box["coordinate"][1]

            json_results.append(
                {
                    "image_id": image_id,
                    "label": box["label"],
                    "bbox": [x, y, w, h],
                    "score": float(box["score"]),
                }
            )

        layout_for_lp = Layout()

        # for score, box, label in zip(scores, boxes, labels):
        for res in json_results:
            score = res["score"]
            label = res["label"]
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

        output = self.model.predict(input=image, batch_size=1)
        layout = self.gather_output(output[0])

        return layout

    def image_loader(self, image):
        return image

    def display_with_paddle(self, res):
        """
        Internal code, used for debugging and visualization using YOLOv10's built-in plotting function.
        res = outputs[0]
        """
        pass
