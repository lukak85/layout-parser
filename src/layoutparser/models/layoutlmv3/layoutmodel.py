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

from .catalog import MODEL_CATALOG
from .ditod.config import add_vit_config
from ..base_layoutmodel import BaseLayoutModel
from ...elements import Rectangle, TextBlock, Layout
from ...file_utils import is_detectron2_available

if is_detectron2_available():
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.engine import default_setup

__all__ = ["LayoutLMv3LayoutModel"]


class LayoutLMv3LayoutModel(BaseLayoutModel):
    """Create a Detectron2-based Layout Detection Model

    Args:
        config_path (:obj:`str`):
            The path to the configuration file.
        model_path (:obj:`str`, None):
            The path to the saved weights of the model.
            If set, overwrite the weights in the configuration file.
            Defaults to `None`.
        label_map (:obj:`dict`, optional):
            The map from the model prediction (ids) to real
            word labels (strings). If the config is from one of the supported
            datasets, Layout Parser will automatically initialize the label_map.
            Defaults to `None`.
        device(:obj:`str`, optional):
            Whether to use cuda or cpu devices. If not set, LayoutParser will
            automatically determine the device to initialize the models on.
        extra_config (:obj:`list`, optional):
            Extra configuration passed to the Detectron2 model
            configuration. The argument will be used in the `merge_from_list
            <https://detectron2.readthedocs.io/modules/config.html
            #detectron2.config.CfgNode.merge_from_list>`_ function.
            Defaults to `[]`.

    Examples::
        >>> import layoutparser as lp
        >>> model = lp.LayoutLMv3LayoutModel('lp://HJDataset/faster_rcnn_R_50_FPN_3x/config')
        >>> model.detect(image)

    """

    DEPENDENCIES = ["detectron2"]
    DETECTOR_NAME = "layoutlmv3"
    def __init__(
        self,
        model_path,
        yaml_path="cascade_layoutlmv3.yaml",
        args=None,
        label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
        extra_config=None,
        enforce_cpu=None,
        device=None,
    ):
        # TODO: currently works with only one GPU, expand to more
        self.cfg = get_cfg()
        # add_coat_config(cfg)
        add_vit_config(self.cfg)
        self.cfg.merge_from_file(os.path.abspath(yaml_path))
        # self.cfg.merge_from_list(None)
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.freeze()
        default_setup(self.cfg, args)

        self.args = args

        self.label_map = label_map

        #self.model = MyTrainer.build_model(self.cfg)
        self._create_model()

    MODEL_CATALOG = MODEL_CATALOG


    def _create_model(self):
        self.model = DefaultPredictor(self.cfg)

    def gather_output(self, outputs):

        instance_pred = outputs["instances"].to("cpu")

        layout = Layout()
        scores = instance_pred.scores.tolist()
        boxes = instance_pred.pred_boxes.tensor.tolist()
        labels = instance_pred.pred_classes.tolist()

        for score, box, label in zip(scores, boxes, labels):
            if score < 0.1:
                continue

            x_1, y_1, x_2, y_2 = box


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

        image = self.image_loader(path)
        outputs = self.model(image)
        layout = self.gather_output(outputs)
        return layout

    def image_loader(self, path, to_rgb=True):
        img = cv2.imread(path)
        if to_rgb:
            img = img[..., ::-1]
        return img

