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

import json
import os

from PIL import Image

from dots_ocr.model.inference import inference_with_vllm
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_ocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from .catalog import MODEL_CATALOG
from ..base_layoutmodel import BaseLayoutModel
from ...elements import Rectangle, TextBlock, Layout

__all__ = ["DotsOCRLayoutModel"]


class DotsOCRLayoutModel(BaseLayoutModel):
    """Create a Detectron2-based Layout Detection Model

    Examples::
        >>> import layoutparser as lp
        >>> model = lp.DocstrumLayoutModel()
        >>> model.detect(image)

    """

    DEPENDENCIES = []
    DETECTOR_NAME = "dotsocr"
    MODEL_CATALOG = MODEL_CATALOG

    def __init__(
        self,
        protocol='http',
        ip='localhost',
        port=8000,
        model_name='model',
        temperature=0.1,
        top_p=1.0,
        max_completion_tokens=16384,
        num_thread=16,
        dpi = 200,
        min_pixels=None,
        max_pixels=None,
        use_hf=False,
        fitz_preprocess=True,
        label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}
    ):
        self.protocol=protocol
        self.ip=ip
        self.port=port
        self.model_name=model_name
        self.temperature=temperature
        self.top_p=top_p
        self.max_completion_tokens=max_completion_tokens
        self.num_thread=num_thread
        self.dpi=dpi
        self.min_pixels=min_pixels
        self.max_pixels=max_pixels
        self.use_hf=use_hf
        self.label_map=label_map
        self.fitz_preprocess=fitz_preprocess

        # print(f"use vllm model, num_thread will be set to {self.num_thread}")
        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS


    def gather_output(self, data):
        layout = Layout()

        for item in data:
            x_1, y_1, x_2, y_2 = [v /2.085 for v in item["bbox"]]

            label = item["category"]
            label = self.label_map.get(label, label)

            cur_block = TextBlock(
                Rectangle(x_1, y_1, x_2, y_2), type=label, score=1
            )
            layout.append(cur_block)

        return layout

    def _inference_with_vllm(self, image, prompt):
        response = inference_with_vllm(
            image,
            prompt,
            model_name=self.model_name,
            protocol=self.protocol,
            ip=self.ip,
            port=self.port,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
        )
        return response

    def get_prompt(self, prompt_mode):
        return dict_promptmode_to_prompt[prompt_mode]

    def _parse_single_image(
            self,
            origin_image,
            prompt_mode,
            source="image",
            fitz_preprocess=False,
    ):
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if min_pixels is not None: assert min_pixels >= MIN_PIXELS, f"min_pixels should >= {MIN_PIXELS}"
        if max_pixels is not None: assert max_pixels <= MAX_PIXELS, f"max_pixels should <= {MAX_PIXELS}"

        if source == 'image' and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
        prompt = self.get_prompt(prompt_mode)

        response = self._inference_with_vllm(image, prompt)


        return response

    def parse_image(self, input_path, prompt_mode, fitz_preprocess=False):
        origin_image = fetch_image(input_path)
        result = self._parse_single_image(origin_image, prompt_mode, fitz_preprocess=fitz_preprocess)
        return result

    def detect(self, path):
        results = self.parse_image(path, "prompt_layout_all_en", fitz_preprocess=self.fitz_preprocess)

        return self.gather_output(json.loads(results))

    def image_loader(self, image):
        return Image.open(image)

