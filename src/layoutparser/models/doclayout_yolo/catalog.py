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
    "DocStructBench": {
        0: "title",
        1: "plain text",
        2: "abandon",
        3: "figure",
        4: "figure_caption",
        5: "table",
        6: "table_caption",
        7: "table_footnote",
        8: "isolate_formula",
        9: "formula_caption"
    },
    "Glasana": {
        0: "Abandon",
        1: "Advertisement",
        2: "Author",
        3: "Byline",
        4: "Caption",
        5: "CaptionByline",
        6: "Dateline",
        7: "Deck",
        8: "Dropcap",
        9: "EditNote",
        10: "FigByline",
        11: "Figure",
        12: "Footer",
        13: "Footnote",
        14: "Form",
        15: "Header",
        16: "Headline",
        17: "Kicker",
        18: "Literary",
        19: "Literature",
        20: "MarginNote",
        21: "OrderedList",
        22: "PageNum",
        23: "Paragraph",
        24: "Question",
        25: "Quote",
        26: "Section",
        27: "Subhead",
        28: "Subsubhead",
        29: "TOC",
        30: "Translator",
        31: "UnorderedList"
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
