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

__version__ = "0.3.4"

import sys

from .file_utils import (
    _LazyModule,
    is_detectron2_available,
    is_doclayout_yolo_available,
    is_layoutlmv3_available,
    is_dit_available,
    is_nemotron_available,
    is_vgt_available,
    is_ppdoclayoutv3_available,
    is_rfdetr_available,
    is_detr_available,
    is_swindocseg_available,
    is_dotsocr_available,
    is_docstrum_available,
    is_recursive_xycut_available,
    is_rlsa_available,
    is_paddle_available,
    is_effdet_available,
    is_pytesseract_available,
    is_gcv_available, is_rfdetr_available,
)

_import_structure = {
    "elements": ["Interval", "Rectangle", "Quadrilateral", "TextBlock", "Layout"],
    "visualization": ["draw_box", "draw_text"],
    "io": ["load_json", "load_dict", "load_csv", "load_dataframe", "load_pdf"],
    "file_utils": [
        "is_torch_available",
        "is_torch_cuda_available",
        "is_detectron2_available",
        "is_doclayout_yolo_available",
        "is_layoutlmv3_available",
        "is_dit_available",
        "is_nemotron_available",
        "is_vgt_available",
        "is_dotsocr_available",
        "is_docstrum_available",
        "is_recursive_xycut_available",
        "is_rlsa_available",
        "is_ppdoclayoutv3_available",
        "is_rfdetr_available",
        "is_detr_available",
        "is_swindocseg_available",
        "is_paddle_available",
        "is_pytesseract_available",
        "is_gcv_available",
        "requires_backends",
    ],
    "tools": [
        "generalized_connected_component_analysis_1d",
        "simple_line_detection",
        "group_textblocks_based_on_category",
    ],
}

_import_structure["models"] = ["AutoLayoutModel"]

if is_detectron2_available():
    _import_structure["models.detectron2"] = ["Detectron2LayoutModel"]

if is_doclayout_yolo_available():
    _import_structure["models.doclayout_yolo"] = ["DocLayoutYOLOLayoutModel"]

if is_layoutlmv3_available():
    _import_structure["models.layoutlmv3"] = ["LayoutLMv3LayoutModel"]

if is_dit_available():
    _import_structure["models.dit"] = ["DiTLayoutModel"]

if is_nemotron_available():
    _import_structure["models.nemotron"] = ["NemotronLayoutModel"]

if is_vgt_available():
    _import_structure["models.vgt"] = ["VGTLayoutModel"]

if is_ppdoclayoutv3_available():
    _import_structure["models.ppdoclayoutv3"] = ["PPDocLayoutV3LayoutModel"]

if is_rfdetr_available():
    _import_structure["models.rfdetr"] = ["RFDETRLayoutModel"]

if is_detr_available():
    _import_structure["models.detr"] = ["DETRLayoutModel"]

if is_swindocseg_available():
    _import_structure["models.swindocseg"] = ["SwinDocSegLayoutModel"]

if is_dotsocr_available():
    _import_structure["models.dotsocr"] = ["DotsOCRLayoutModel"]

if is_docstrum_available():
    _import_structure["models.docstrum"] = ["DocstrumLayoutModel"]

if is_recursive_xycut_available():
    _import_structure["models.recursive_xycut"] = ["RecursiveXYCutLayoutModel"]

if is_rlsa_available():
    _import_structure["models.rlsa"] = ["RLSALayoutModel"]

if is_paddle_available():
    _import_structure["models.paddledetection"] = ["PaddleDetectionLayoutModel"]

if is_effdet_available():
    _import_structure["models.effdet"] = ["EfficientDetLayoutModel"]

if is_pytesseract_available():
    _import_structure["ocr.tesseract_agent"] = [
        "TesseractAgent",
        "TesseractFeatureType",
    ]

if is_gcv_available():
    _import_structure["ocr.gcv_agent"] = ["GCVAgent", "GCVFeatureType"]

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
    module_spec=__spec__,
    extra_objects={"__version__": __version__},
)
