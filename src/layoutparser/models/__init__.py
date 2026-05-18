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

from .. import is_doclayout_yolo_available
from .. import is_layoutlmv3_available
from .. import is_dit_available
from .. import is_nemotron_available
from .. import is_vgt_available
from .. import is_dotsocr_available
from .. import is_docstrum_available
from .. import is_recursive_xycut_available
from .. import is_rlsa_available
from .. import is_ppdoclayoutv3_available
from .. import is_rfdetr_available
from .. import is_swindocseg_available

from .detectron2.layoutmodel import Detectron2LayoutModel
from .paddledetection.layoutmodel import PaddleDetectionLayoutModel
from .effdet.layoutmodel import EfficientDetLayoutModel
from .auto_layoutmodel import AutoLayoutModel

if is_doclayout_yolo_available():
    from .doclayout_yolo.layoutmodel import DocLayoutYOLOLayoutModel
if is_layoutlmv3_available():
    from .layoutlmv3.layoutmodel import LayoutLMv3LayoutModel
if is_dit_available():
    from .dit.layoutmodel import DiTLayoutModel
if is_nemotron_available():
    from .nemotron.layoutmodel import NemotronLayoutModel
if is_vgt_available():
    from .vgt.layoutmodel import VGTLayoutModel
if is_dotsocr_available():
    from .dotsocr.layoutmodel import DotsOCRLayoutModel
if is_docstrum_available():
    from .docstrum.layoutmodel import DocstrumLayoutModel
if is_recursive_xycut_available():
    from .recursive_xycut.layoutmodel import RecursiveXYCutLayoutModel
if is_rlsa_available():
    from .rlsa.layoutmodel import RLSALayoutModel
if is_ppdoclayoutv3_available():
    from .ppdoclayoutv3.layoutmodel import PPDocLayoutV3LayoutModel
if is_rfdetr_available():
    from .rfdetr.layoutmodel import RFDETRLayoutModel
if is_swindocseg_available():
    from .swindocseg.layoutmodel import SwinDocSegLayoutModel