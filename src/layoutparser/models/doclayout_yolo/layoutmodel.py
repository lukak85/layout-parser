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
import cv2
import torch

from doclayout_yolo import YOLOv10
from torch.backends.mkl import verbose

from .catalog import MODEL_CATALOG, LABEL_MAP_CATALOG
from ..base_layoutmodel import BaseLayoutModel
from ...elements import Rectangle, TextBlock, Layout

__all__ = ["DocLayoutYOLOLayoutModel"]


class DocLayoutYOLOLayoutModel(BaseLayoutModel):

    DEPENDENCIES = ["doclayout_yolo"]
    DETECTOR_NAME = "doclayout_yolo"
    MODEL_CATALOG = MODEL_CATALOG

    def __init__(
        self,
        model,
        imgsz=1024,
        conf=0.2,
        debug=False,
        label_map=None,
        verbose=False,
    ):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        self.model = model
        self.imgsz = imgsz
        self.conf = conf

        self.debug = debug

        if label_map is None:
            label_map = LABEL_MAP_CATALOG["PubLayNet"]
        else:
            label_map = LABEL_MAP_CATALOG[label_map]

        self.label_map = label_map

        self.verbose = verbose

        self._create_model()

    def _create_model(self):
        self.model = YOLOv10(self.model, verbose=self.verbose)  # load an official model

    def gather_output(self, yolo_result, image_id, img_w, img_h):
        coco_results = []

        boxes = yolo_result[0].boxes

        xywhn = boxes.xywhn.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for (xc, yc, w, h), score, cls in zip(xywhn, confs, clss):
            x = (xc - w / 2) * img_w
            y = (yc - h / 2) * img_h
            w = w * img_w
            h = h * img_h

            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id": cls,
                    "bbox": [x, y, w, h],
                    "score": float(score),
                }
            )

        layout_for_lp = Layout()

        # for score, box, label in zip(scores, boxes, labels):
        for res in coco_results:
            score = res["score"]
            label = self.label_map.get(res["category_id"], res["category_id"])
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

        image = self.image_loader(image)
        outputs = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
        )
        # Read image dimensions from file
        from PIL import Image

        img = Image.open(image)
        width, height = img.size

        # Debugging code, visualize the detection results using YOLOv10's built-in plotting function.
        if self.debug:
            self.display_with_yolo(outputs[0])

        layout = self.gather_output(outputs, 1, width, height)

        return layout

    def image_loader(self, image):
        return image

    def display_with_yolo(self, res):
        """
        Internal code, used for debugging and visualization using YOLOv10's built-in plotting function.
        res = outputs[0]
        """

        annotated_frame = res.plot(pil=True, line_width=5, font_size=20)
        cv2.imshow("annotated_frame", annotated_frame)
        # Wait until a key is pressed and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        quit()
