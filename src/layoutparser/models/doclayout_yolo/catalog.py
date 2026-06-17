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

from iopath.common.file_io import PathHandler

from ..base_catalog import PathManager

MODEL_CATALOG = {
    "DocLayNet": {
        "doclayout_yolo_docstructbench_imgsz1024": "https://huggingface.co/juliozhao/DocLayout-YOLO-DocLayNet-from_scratch",
        "doclayout_yolo_docstructbench_imgsz10242": "https://huggingface.co/juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained",
    },
    "D4LA": {
        "doclayout_yolo_docstructbench_imgsz1024": "https://huggingface.co/juliozhao/DocLayout-YOLO-D4LA-from_scratch",
        "doclayout_yolo_docstructbench_imgsz10242": "https://huggingface.co/juliozhao/DocLayout-YOLO-D4LA-Docsynth300K_pretrained",
    },
    "DocSynth300K": {
        "doclayout_yolo_docsynth300k_imgsz1600": "https://huggingface.co/juliozhao/DocLayout-YOLO-DocSynth300K-pretrain",
    },
}

# fmt: off
LABEL_MAP_CATALOG = {
    "DocLayNet": {
        0: "Caption",
        1: "Footnote",
        2: "Formula",
        3: "List-item",
        4: "Page-footer",
        5: "Page-header",
        6: "Picture",
        7: "Section-header",
        8: "Table",
        9: "Text",
        10: "Title"
    },
    "D4LA": {
        0: "DocTitle",
        1: "ParaTitle",
        2: "ParaText",
        3: "ListText",
        4: "RegionTitle",
        5: "Date",
        6: "LetterHead",
        7: "LetterDear",
        8: "LetterSign",
        9: "Question",
        10: "OtherText",
        11: "RegionKV",
        12: "RegionList",
        13: "Abstract",
        14: "Author",
        15: "TableName",
        16: "Table",
        17: "Figure",
        18: "FigureName",
        19: "Equation",
        20: "Reference",
        21: "Footer",
        22: "PageHeader",
        23: "PageFooter",
        24: "Number",
        25: "Catalog",
        26: "PageNumber"
    },
    "DocSynth300K": {
        0: "QR code",
        1: "advertisement",
        2: "algorithm",
        3: "answer",
        4: "author",
        5: "barcode",
        6: "bill",
        7: "blank",
        8: "bracket",
        9: "breakout",
        10: "byline",
        11: "caption",
        12: "catalogue",
        13: "chapter title",
        14: "code",
        15: "correction",
        16: "credit",
        17: "dateline",
        18: "drop cap",
        19: "editor's note",
        20: "endnote",
        21: "examinee information",
        22: "fifth-level title",
        23: "figure",
        24: "first-level question number",
        25: "first-level title",
        26: "flag",
        27: "folio",
        28: "footer",
        29: "footnote",
        30: "formula",
        31: "fourth-level section title",
        32: "fourth-level title",
        33: "header",
        34: "headline",
        35: "index",
        36: "inside",
        37: "institute",
        38: "jump line",
        39: "kicker",
        40: "lead",
        41: "marginal note",
        42: "matching",
        43: "mugshot",
        44: "option",
        45: "ordered list",
        46: "other question number",
        47: "page number",
        48: "paragraph",
        49: "part",
        50: "play",
        51: "poem",
        52: "reference",
        53: "sealing line",
        54: "second-level question number",
        55: "second-level title",
        56: "section",
        57: "section title",
        58: "sidebar",
        59: "sub section title",
        60: "subhead",
        61: "subsub section title",
        62: "supplementary note",
        63: "table",
        64: "table caption",
        65: "table note",
        66: "teasers",
        67: "third-level question number",
        68: "third-level title",
        69: "title",
        70: "translator",
        71: "underscore",
        72: "unordered list",
        73: "weather forecast",
    },
    "Glasana": {
        0: "Abandon",
        1: "Advertisement",
        2: "Author",
        3: "Byline",
        4: "Caption",
        5: "Dateline",
        6: "Deck",
        7: "Dropcap",
        8: "EditNote",
        9: "FigByline",
        10: "Figure",
        11: "Footer",
        12: "Footnote",
        13: "Header",
        14: "Headline",
        15: "Kicker",
        16: "Literary",
        17: "Literature",
        18: "MarginNote",
        19: "OrderedList",
        20: "PageNum",
        21: "Paragraph",
        22: "Question",
        23: "Quote",
        24: "Section",
        25: "Subhead",
        26: "Subsubhead",
        27: "TOC",
        28: "Translator",
        29: "UnorderedList"
    }
}
# fmt: on


class LayoutParserDocLayoutYOLOModelHandler(PathHandler):
    """
    Resolve anything that's in LayoutParser model zoo.
    """

    PREFIX = "lp://doclayout-yolo/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path, **kwargs):
        model_name = path[len(self.PREFIX) :]

        dataset_name, *model_name, data_type = model_name.split("/")

        if data_type == "weight":
            model_url = MODEL_CATALOG[dataset_name]["/".join(model_name)]
        else:
            raise ValueError(f"Unknown data_type {data_type}")
        return PathManager.get_local_path(model_url, **kwargs)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager.register_handler(LayoutParserDocLayoutYOLOModelHandler())
