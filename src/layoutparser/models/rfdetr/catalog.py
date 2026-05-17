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
}

CONFIG_CATALOG = {

}

# fmt: off
LABEL_MAP_CATALOG = {
    "PubLayNet": {
        0: 'title',
        1: 'plain text',
        2: 'abandon',
        3: 'figure',
        4: 'figure_caption',
        5: 'table',
        6: 'table_caption',
        7: 'table_footnote',
        8: 'isolate_formula',
        9: 'formula_caption'},
    "Glasana": {
        0: 'headline',
        1: 'paragraph',
        2: 'admonition',
        3: 'figure',
        4: 'caption',
        5: 'figure',
        6: 'caption',
        7: 'caption',
        8: 'figure',
        9: 'caption',
    }
}
# fmt: on


class LayoutParserDocLayoutYOLOModelHandler(PathHandler):
    """
    Resolve anything that's in LayoutParser model zoo.
    """

    PREFIX = "lp://detectron2/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path, **kwargs):
        model_name = path[len(self.PREFIX) :]

        dataset_name, *model_name, data_type = model_name.split("/")

        if data_type == "weight":
            model_url = MODEL_CATALOG[dataset_name]["/".join(model_name)]
        elif data_type == "config":
            model_url = CONFIG_CATALOG[dataset_name]["/".join(model_name)]
        else:
            raise ValueError(f"Unknown data_type {data_type}")
        return PathManager.get_local_path(model_url, **kwargs)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager.register_handler(LayoutParserDocLayoutYOLOModelHandler())
