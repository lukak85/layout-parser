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

import os
import cv2
import torch

from .catalog import MODEL_CATALOG, LABEL_MAP_CATALOG
from ..base_layoutmodel import BaseLayoutModel
from ...elements import Rectangle, TextBlock, Layout
from ...file_utils import is_detectron2_available

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detectron2.projects.deeplab import add_deeplab_config
from maskdino.config import add_maskformer2_config

if is_detectron2_available():
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.engine import default_setup


__all__ = ["SwinDocSegLayoutModel"]


class SwinDocSegLayoutModel(BaseLayoutModel):
    DEPENDENCIES = ["detectron2"]
    DETECTOR_NAME = "swindocseg"
    MODEL_CATALOG = MODEL_CATALOG

    def __init__(
        self,
        model_path,
        yaml_path="cascade_dit_base.yaml",
        args=None,
        label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
        score_threshold=0.1,
    ):
        # Step 1: instantiate config
        self.cfg = get_cfg()
        add_deeplab_config(self.cfg)
        add_maskformer2_config(self.cfg)
        self.cfg.merge_from_file(os.path.abspath(yaml_path))

        # Step 2: add model weights URL to config
        self.cfg.MODEL.WEIGHTS = model_path


        if label_map is None:
            label_map = LABEL_MAP_CATALOG["PubLayNet"]
        else:
            label_map = LABEL_MAP_CATALOG[label_map]

        self.label_map = label_map

        self.score_threshold = score_threshold

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg.MODEL.DEVICE = device

        self.cfg.freeze()

        default_setup(self.cfg, args)

        self._create_model()

    def _create_model(self):
        self.model = DefaultPredictor(self.cfg)

    def gather_output(self, outputs):

        instance_pred = outputs["instances"].to("cpu")

        layout = Layout()
        scores = instance_pred.scores.tolist()
        boxes = instance_pred.pred_boxes.tensor.tolist()
        labels = instance_pred.pred_classes.tolist()

        for score, box, label in zip(scores, boxes, labels):
            if score < self.score_threshold:
                continue

            x_1, y_1, x_2, y_2 = box


            label = self.label_map.get(label, label)

            cur_block = TextBlock(
                Rectangle(x_1, y_1, x_2, y_2), type=label, score=score
            )
            layout.append(cur_block)

        return layout

    def detect(self, path):
        image = self.image_loader(path, False)
        outputs = self.model(image)
        layout = self.gather_output(outputs)
        return layout

    def image_loader(self, path, to_rgb=True):
        img = cv2.imread(path)
        if to_rgb:
            img = img[..., ::-1]
        return img
