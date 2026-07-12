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

from pathlib import Path

import PIL.Image

from .catalog import MODEL_CATALOG, LABEL_MAP_CATALOG
from ..base_layoutmodel import BaseLayoutModel
from ...elements import Rectangle, TextBlock, Layout

__all__ = ["DETRLayoutModel"]

_DEFAULT_MODEL = "cmarkea/detr-layout-detection"


class DETRLayoutModel(BaseLayoutModel):
    """Create a DETR-based document layout detection model.

    Wraps the `cmarkea/detr-layout-detection` model (a DETR fine-tuned on
    DocLayNet, exposed through HuggingFace ``transformers``).

    Args:
        model_name (:obj:`str`, optional):
            The HuggingFace model id (or local path) to load. Defaults to
            ``cmarkea/detr-layout-detection``.
        label_map (:obj:`str` or :obj:`dict`, optional):
            Either the name of a label map in the catalog (e.g. ``"DocLayNet"``)
            or an explicit ``{id: name}`` mapping. Defaults to ``"DocLayNet"``.
        score_threshold (:obj:`float`, optional):
            Minimum confidence for a detection to be kept. Defaults to ``0.4``.
        device (:obj:`str`, optional):
            ``"cuda"`` or ``"cpu"``. Auto-detected when not set.

    Examples::
        >>> import layoutparser as lp
        >>> model = lp.DETRLayoutModel()
        >>> model.detect(image)
    """

    DEPENDENCIES = ["detr"]
    DETECTOR_NAME = "detr"
    MODEL_CATALOG = MODEL_CATALOG

    def __init__(
        self,
        model_name=_DEFAULT_MODEL,
        label_map=None,
        score_threshold=0.4,
        device=None,
    ):
        import torch
        from transformers import AutoImageProcessor
        from transformers.models.detr import DetrForSegmentation

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._img_proc = AutoImageProcessor.from_pretrained(model_name)
        self._model = DetrForSegmentation.from_pretrained(model_name).to(device)
        self._model.eval()

        if label_map is None:
            label_map = LABEL_MAP_CATALOG["DocLayNet"]
        elif isinstance(label_map, str):
            label_map = LABEL_MAP_CATALOG[label_map]
        self.label_map = label_map

        self.score_threshold = score_threshold

    def gather_output(self, detection):
        layout = Layout()

        scores = detection["scores"].tolist()
        labels = detection["labels"].tolist()
        boxes = detection["boxes"].tolist()

        for score, label, box in zip(scores, labels, boxes):
            x_1, y_1, x_2, y_2 = box
            label = self.label_map.get(label, label)
            layout.append(
                TextBlock(
                    Rectangle(x_1, y_1, x_2, y_2), type=label, score=score
                )
            )

        return layout

    def detect(self, image):
        """Detect the layout of a given image.

        Args:
            image (:obj:`str`, :obj:`~pathlib.Path` or `PIL.Image`): The input
                image (path or PIL image) to detect.

        Returns:
            :obj:`~layoutparser.Layout`: The detected layout of the input image
        """
        import torch

        img = self.image_loader(image)

        inputs = self._img_proc(images=img, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self._model(**inputs)

        detections = self._img_proc.post_process_object_detection(
            output,
            threshold=self.score_threshold,
            target_sizes=[img.size[::-1]],
        )

        return self.gather_output(detections[0])

    def image_loader(self, image) -> PIL.Image.Image:
        if isinstance(image, (str, Path)):
            image = PIL.Image.open(image)
        return image.convert("RGB")