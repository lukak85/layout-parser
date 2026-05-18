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

# Some code are adapted from
# https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py

from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
import sys
import os
import logging
import importlib.util
from types import ModuleType

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

# If LAYOUTPARSER_BACKEND is set, only the specified backend(s) will report as
# available. Comma-separated list, e.g. "layoutlmv3" or "doclayout_yolo,vgt".
# This prevents backends that share a dependency (e.g. detectron2) from all
# appearing available when only one is properly configured.
_allowed_backends = os.environ.get("LAYOUTPARSER_BACKEND", None)
if _allowed_backends is not None:
    _allowed_backends = {b.strip().lower() for b in _allowed_backends.split(",")}
    logger.debug(f"LAYOUTPARSER_BACKEND restricts available backends to: {_allowed_backends}")


def _is_backend_allowed(name: str) -> bool:
    """Return True if this backend is allowed by LAYOUTPARSER_BACKEND (or if the env var is unset)."""
    if _allowed_backends is None:
        return True
    return name.lower() in _allowed_backends

###########################################
############ Layout Model Deps ############
###########################################

_torch_available = importlib.util.find_spec("torch") is not None
try:
    _torch_version = importlib_metadata.version("torch")
    logger.debug(f"PyTorch version {_torch_version} available.")
except importlib_metadata.PackageNotFoundError:
    _torch_available = False

_detectron2_available = importlib.util.find_spec("detectron2") is not None
try:
    _detectron2_version = importlib_metadata.version("detectron2")
    logger.debug(f"Detectron2 version {_detectron2_version} available")
except importlib_metadata.PackageNotFoundError:
    _detectron2_available = False

_doclayout_yolo_available = _is_backend_allowed("doclayout_yolo") and importlib.util.find_spec("doclayout_yolo") is not None
try:
    _doclayout_yolo_version = importlib_metadata.version("doclayout_yolo")
    logger.debug(f"DocLayout-YOLO version {_doclayout_yolo_version} available")
except importlib_metadata.PackageNotFoundError:
    _doclayout_yolo_available = False

_layoutlmv3_available = _is_backend_allowed("layoutlmv3") and importlib.util.find_spec("detectron2") is not None
try:
    _layoutlmv3_version = importlib_metadata.version("detectron2")
    logger.debug(f"LayoutLMv3 version {_layoutlmv3_version} available")
except importlib_metadata.PackageNotFoundError:
    _layoutlmv3_available = False

_dit_available = _is_backend_allowed("dit") and importlib.util.find_spec("detectron2") is not None
try:
    _dit_version = importlib_metadata.version("detectron2")
    logger.debug(f"DiT version {_dit_version} available")
except importlib_metadata.PackageNotFoundError:
    _dit_available = False

_nemotron_available = _is_backend_allowed("nemotron") #and importlib.util.find_spec("nemotron_page_elements_v3") is not None
try:
    #_nemotron_version = importlib_metadata.version("nemotron-page-elements-v3") # TODO: figure this out
    _nemotron_version = "1.0.0"
    logger.debug(f"Nemotron version {_nemotron_version} available")
except importlib_metadata.PackageNotFoundError:
    _nemotron_available = False

_vgt_available = _is_backend_allowed("vgt") and importlib.util.find_spec("detectron2") is not None
try:
    _vgt_version = importlib_metadata.version("detectron2")
    logger.debug(f"VGT version {_vgt_version} available")
except importlib_metadata.PackageNotFoundError:
    _vgt_available = False


_dotsocr_available = _is_backend_allowed("dotsocr") and importlib.util.find_spec("dots_ocr") is not None
try:
    _dotsocr_version = importlib_metadata.version("dots_ocr")
    logger.debug(f"dots.ocr version {_dotsocr_version} available")
except importlib_metadata.PackageNotFoundError:
    _dotsocr_available = False

_docstrum_available = _is_backend_allowed("docstrum") and importlib.util.find_spec("shapely") is not None
try:
    _docstrum_version = importlib_metadata.version("shapely")
    logger.debug(f"Docstrum (shapely) version {_docstrum_version} available")
except importlib_metadata.PackageNotFoundError:
    _docstrum_available = False

_recursive_xycut_available = _is_backend_allowed("recursive_xycut") and importlib.util.find_spec("numpy") is not None
try:
    _recursive_xycut_version = importlib_metadata.version("numpy")
    logger.debug(f"RecursiveXYCut (numpy) version {_recursive_xycut_version} available")
except importlib_metadata.PackageNotFoundError:
    _recursive_xycut_available = False

_rlsa_available = _is_backend_allowed("rlsa") and importlib.util.find_spec("numpy") is not None
try:
    _rlsa_version = importlib_metadata.version("numpy")
    logger.debug(f"RLSA (numpy) version {_rlsa_version} available")
except importlib_metadata.PackageNotFoundError:
    _rlsa_available = False

_ppdoclayoutv3_available = _is_backend_allowed("ppdoclayoutv3") and importlib.util.find_spec("paddleocr") is not None
try:
    _ppdoclayoutv3_version = importlib_metadata.version("paddleocr")
    logger.debug(f"PP-DocLayoutV3 version {_ppdoclayoutv3_version} available")
except importlib_metadata.PackageNotFoundError:
    _ppdoclayoutv3_available = False

_rfdetr_available = _is_backend_allowed("rfdetr") and importlib.util.find_spec("rfdetr_doclayout") is not None
try:
    _rfdetr_version = importlib_metadata.version("rfdetr_doclayout")
    logger.debug(f"RF-DETR version {_rfdetr_version} available")
except importlib_metadata.PackageNotFoundError:
    _rfdetr_available = False


_swindocseg_available = _is_backend_allowed("swindocseg") and importlib.util.find_spec("detectron2") is not None
try:
    _swindocseg_version = importlib_metadata.version("detectron2")
    logger.debug(f"SwinDocSegmenter version {_swindocseg_version} available")
except importlib_metadata.PackageNotFoundError:
    _swindocseg_available = False

_paddle_available = _is_backend_allowed("paddle") and importlib.util.find_spec("paddle") is not None
try:
    # The name of the paddlepaddle library:
    # Install name: pip install paddlepaddle
    # Import name: import paddle
    _paddle_version = importlib_metadata.version("paddlepaddle")
    logger.debug(f"Paddle version {_paddle_version} available.")
except importlib_metadata.PackageNotFoundError:
    _paddle_available = False

_effdet_available = _is_backend_allowed("effdet") and importlib.util.find_spec("effdet") is not None
try:
    _effdet_version = importlib_metadata.version("effdet")
    logger.debug(f"Effdet version {_effdet_version} available.")
except importlib_metadata.PackageNotFoundError:
    _effdet_version = False

###########################################
############## OCR Tool Deps ##############
###########################################

_pytesseract_available = importlib.util.find_spec("pytesseract") is not None
try:
    _pytesseract_version = importlib_metadata.version("pytesseract")
    logger.debug(f"Pytesseract version {_pytesseract_version} available.")
except importlib_metadata.PackageNotFoundError:
    _pytesseract_available = False

try:
    _gcv_available = importlib.util.find_spec("google.cloud.vision") is not None
    try:
        _gcv_version = importlib_metadata.version(
            "google-cloud-vision"
        )  # This is slightly different
        logger.debug(f"Google Cloud Vision Utils version {_gcv_version} available.")
    except importlib_metadata.PackageNotFoundError:
        _gcv_available = False
except ModuleNotFoundError:
    _gcv_available = False


def is_torch_available():
    return _torch_available


def is_torch_cuda_available():
    if is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False


def is_detectron2_available():
    return _detectron2_available


def is_doclayout_yolo_available():
    return _doclayout_yolo_available


def is_layoutlmv3_available():
    return _layoutlmv3_available


def is_dit_available():
    return _dit_available


def is_nemotron_available():
    return _nemotron_available


def is_vgt_available():
    return _vgt_available

def is_dotsocr_available():
    return _dotsocr_available


def is_docstrum_available():
    return _docstrum_available


def is_recursive_xycut_available():
    return _recursive_xycut_available


def is_rlsa_available():
    return _rlsa_available


def is_ppdoclayoutv3_available():
    return _ppdoclayoutv3_available


def is_rfdetr_available():
    return _rfdetr_available

def is_swindocseg_available():
    return _swindocseg_available


def is_paddle_available():
    return _paddle_available


def is_effdet_available():
    return _effdet_available


def is_pytesseract_available():
    return _pytesseract_available


def is_gcv_available():
    return _gcv_available


PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
"""

DETECTRON2_IMPORT_ERROR = """
{0} requires the detectron2 library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md and follow the ones
that match your environment. Typically the following would work for MacOS or Linux CPU machines:
    pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.4#egg=detectron2' 
"""

DOCLAYOUT_YOLO_IMPORT_ERROR = """
{0} requires the doclayout-yolo library but it was not found in your environment. TOOD - add installation instructions
here.
"""

LAYOUTLMV3_IMPORT_ERROR = """
{0} requires the layoutlmft library but it was not found in your environment. TOOD - add installation instructions
here.
"""

DIT_IMPORT_ERROR = """
{0} requires the detectron2 library but it was not found in your environment. TOOD - add installation instructions
here.
"""

NEMOTRON_IMPORT_ERROR = """
{0} requires the nemotron-page-layout3 library but it was not found in your environment. TOOD - add installation instructions
here.
"""

VGT_IMPORT_ERROR = """
{0} requires the detectron2 library but it was not found in your environment. TOOD - add installation instructions
here.
"""

PPDOCLAYOUTV3_IMPORT_ERROR = """
{0} requires the paddleocr library but it was not found in your environment. Install using: 'pip install paddleocr'
"""

RFDETR_IMPORT_ERROR = """
{0} requires the rfdetr_doclayout library but it was not found in your environment. Install using: 'pip install rfdetr-doclayout'
"""

DOTSOCR_IMPORT_ERROR = """
{0} requires the dots-ocr library but it was not found in your environment. TOOD - add installation instructions
here.
"""

DOCSTRUM_IMPORT_ERROR = """
{0} requires the shapely library but it was not found in your environment. You can install it with pip:
`pip install shapely`
"""

PADDLE_IMPORT_ERROR = """
{0} requires the PaddlePaddle library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/PaddlePaddle/Paddle and follow the ones that match your environment.
"""

EFFDET_IMPORT_ERROR = """
{0} requires the effdet library but it was not found in your environment. You can install it with pip:
`pip install effdet`
"""

PYTESSERACT_IMPORT_ERROR = """
{0} requires the PyTesseract library but it was not found in your environment. You can install it with pip:
`pip install pytesseract`
"""

GCV_IMPORT_ERROR = """
{0} requires the Google Cloud Vision Python utils but it was not found in your environment. You can install it with pip:
`pip install google-cloud-vision==1`
"""

BACKENDS_MAPPING = dict(
    [
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("detectron2", (is_detectron2_available, DETECTRON2_IMPORT_ERROR)),
        ("doclayout_yolo", (is_doclayout_yolo_available, DOCLAYOUT_YOLO_IMPORT_ERROR)),
        ("layoutlmv3", (is_layoutlmv3_available, LAYOUTLMV3_IMPORT_ERROR)),
        ("dit", (is_dit_available, DIT_IMPORT_ERROR)),
        ("nemotron", (is_nemotron_available, NEMOTRON_IMPORT_ERROR)),
        ("vgt", (is_vgt_available, VGT_IMPORT_ERROR)),
        ("dotsocr", (is_dotsocr_available, DOTSOCR_IMPORT_ERROR)),
        ("docstrum", (is_docstrum_available, DOCSTRUM_IMPORT_ERROR)),
        ("recursive_xycut", (is_recursive_xycut_available, "")),
        ("rlsa", (is_rlsa_available, "")),
        ("ppdoclayoutv3", (is_ppdoclayoutv3_available, PPDOCLAYOUTV3_IMPORT_ERROR)),
        ("rfdetr", (is_rfdetr_available, RFDETR_IMPORT_ERROR)),
        ("swindocseg", (is_swindocseg_available, "")),
        ("paddle", (is_paddle_available, PADDLE_IMPORT_ERROR)),
        ("effdet", (is_effdet_available, EFFDET_IMPORT_ERROR)),
        ("pytesseract", (is_pytesseract_available, PYTESSERACT_IMPORT_ERROR)),
        ("google-cloud-vision", (is_gcv_available, GCV_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    if not all(BACKENDS_MAPPING[backend][0]() for backend in backends):
        raise ImportError(
            "".join([BACKENDS_MAPPING[backend][1].format(name) for backend in backends])
        )


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Adapted from HuggingFace
    # https://github.com/huggingface/transformers/blob/c37573806ab3526dd805c49cbe2489ad4d68a9d7/src/transformers/file_utils.py#L1990

    def __init__(
        self, name, module_file, import_structure, module_spec=None, extra_objects=None
    ):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + sum(
            import_structure.values(), []
        )
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

        # Following [PEP 366](https://www.python.org/dev/peps/pep-0366/)
        # The __package__ variable should be set
        # https://docs.python.org/3/reference/import.html#__package__
        self.__package__ = self.__name__

    # Needed for autocompletion in an IDE
    def __dir__(self):
        return super().__dir__() + self.__all__

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        return importlib.import_module("." + module_name, self.__name__)

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))
