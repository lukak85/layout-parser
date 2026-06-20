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
    "HJDataset": {
        "faster_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/6icw6at8m28a2ho/model_final.pth?dl=1",
        "mask_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/893paxpy5suvlx9/model_final.pth?dl=1",
        "retinanet_R_50_FPN_3x": "https://www.dropbox.com/s/yxsloxu3djt456i/model_final.pth?dl=1",
    },
    "PubLayNet": {
        "faster_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/dgy9c10wykk4lq4/model_final.pth?dl=1",
        "mask_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/d9fc9tahfzyl6df/model_final.pth?dl=1",
        "mask_rcnn_X_101_32x8d_FPN_3x": "https://www.dropbox.com/s/57zjbwv6gh3srry/model_final.pth?dl=1",
    },
    "PrimaLayout": {
        "mask_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/h7th27jfv19rxiy/model_final.pth?dl=1"
    },
    "NewspaperNavigator": {
        "faster_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/6ewh6g8rqt2ev3a/model_final.pth?dl=1",
    },
    "TableBank": {
        "faster_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/8v4uqmz1at9v72a/model_final.pth?dl=1",
        "faster_rcnn_R_101_FPN_3x": "https://www.dropbox.com/s/6vzfk8lk9xvyitg/model_final.pth?dl=1",
    },
    "MFD": {
        "faster_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/7xel0i3iqpm2p8y/model_final.pth?dl=1",
    },
}

CONFIG_CATALOG = {
    "PubLayNet": {
        "faster_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/f3b12qc4hc0yh4m/config.yml?dl=1",
        "mask_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/u9wbsfwz4y0ziki/config.yml?dl=1",
        "mask_rcnn_X_101_32x8d_FPN_3x": "https://www.dropbox.com/s/nau5ut6zgthunil/config.yaml?dl=1",
    },
}

# fmt: off
LABEL_MAP_CATALOG = {
    "PubLayNet": {
        0: "text",
        1: "title",
        2: "list",
        3: "table",
        4: "figure"
    }
}
# fmt: on


class LayoutParserDetectron2ModelHandler(PathHandler):
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


PathManager.register_handler(LayoutParserDetectron2ModelHandler())
