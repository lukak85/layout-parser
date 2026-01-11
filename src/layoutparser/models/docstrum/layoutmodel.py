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

import math
from typing import Union

import numpy as np
from PIL import Image
from shapely.geometry import Point  # For checking overlap
from shapely.geometry.polygon import Polygon  # For checking overlap

from .catalog import MODEL_CATALOG
from .page import Page
from ..base_layoutmodel import BaseLayoutModel
from ...elements import Rectangle, TextBlock, Layout

__all__ = ["DocstrumLayoutModel"]


class DocstrumLayoutModel(BaseLayoutModel):
    """Create a Detectron2-based Layout Detection Model

    Examples::
        >>> import layoutparser as lp
        >>> model = lp.DocstrumLayoutModel()
        >>> model.detect(image)

    """

    DEPENDENCIES = []
    DETECTOR_NAME = "docstrum"
    MODEL_CATALOG = MODEL_CATALOG

    EPS = 1e-10
    THRESHOLD_POLY_EXAGGERATE = 10  # Unit: Pixel

    def __init__(
        self,
        verbose: bool = False,
        # threshold_angle: float = 30.0,
        threshold_angle: float = 60.0,
        threshold_paralldist: float = 2 * 10,
        # threshold_paralldist: float = 1.7 * 13,
        threshold_perpendist: float = 0.75 * 30,
        # threshold_perpendist: float = 1.7 * 17,
        threshold_overlap: float = 1,
        dots_to_lines: bool = True,
        early_skip_steps: int = 100
    ):
        self.verbose = verbose
        self.threshold_angle = threshold_angle
        self.threshold_paralldist = threshold_paralldist
        self.threshold_perpendist = threshold_perpendist
        self.threshold_overlap = threshold_overlap
        self.dots_to_lines = dots_to_lines
        self.early_skip_steps = early_skip_steps

    def detect(self, image):
        page = Page(image)
        if self.verbose:
            page.show()
        return self.process(page)

    def process(self, page: Page):
        image = page.image.copy()

        if self.verbose:
            print(image.shape)

        lines = page.lines

        group_idx = 0

        # Remove dots (avoid zero-length lines)
        actual_lines = []
        for line in lines:
            if not (line.start.x - line.end.x == 0 and line.start.y - line.end.y == 0):
                actual_lines.append(line)
            # Alternativelly, extend line by 3 pixels in both directions
            elif self.dots_to_lines:
                if line.start.x - line.end.x == 0 and line.start.y - line.end.y == 0:
                    line.start.x = line.start.x - 3
                    line.end.x = line.end.x + 3
                    actual_lines.append(line)

        lines = actual_lines

        # TODO: find out if there's a way to optimize this, as this can get quite large
        # Compare each pair of lines and group them
        for line in lines:
            early_skip = 0
            for line2 in lines:
                if early_skip >= self.early_skip_steps:
                    break

                early_skip += 1

                # Same line, so skip
                if line == line2:
                    continue

                # The non-empty lines are already in the same group, so skip
                if line.group is not None and line2.group is not None and line.group == line2.group:
                    continue

                # PART 1: Angle check
                xoi = line.start.x
                # yoi = image.shape[0] - line.start.y
                yoi = line.start.y
                xfi = line.end.x
                # yfi = image.shape[0] - line.end.y
                yfi = line.end.y

                xoj = line2.start.x
                # yoj = image.shape[0] - line2.start.y
                yoj = line2.start.y
                xfj = line2.end.x
                # yfj = image.shape[0] - line2.end.y
                yfj = line2.end.y

                delta_xi = float(xfi - xoi)
                delta_yi = float(yfi - yoi)
                delta_xj = float(xfj - xoj)
                delta_yj = float(yfj - yoj)

                theta_ij = abs(math.atan2(delta_yj, delta_xj) - math.atan2(delta_yi, delta_xi))

                # Angle too large, skip
                if theta_ij > self.threshold_angle:
                    continue

                # PART 2: Overlap check
                if delta_xj != 0:
                    xaj = (xoi * delta_xi * delta_xj + xoj * delta_yi * delta_yj + delta_xj * delta_yi * (yoi - yoj)) / \
                          (delta_yi * delta_yj + delta_xi * delta_xj)
                    yaj = (delta_yj / delta_xj) * (xaj - xoj) + yoj

                    xbj = (xfi * delta_xi * delta_xj + xfj * delta_yi * delta_yj + delta_xj * delta_yi * (yfi - yfj)) / \
                          (delta_yi * delta_yj + delta_xi * delta_xj)
                    ybj = (delta_yj / delta_xj) * (xbj - xfj) + yfj

                else:
                    yaj = (yoi * delta_yi * delta_yj + yoj * delta_xi * delta_xj + delta_yj * delta_xi * (xoi - xoj)) / \
                          (delta_xi * delta_xj + delta_yi * delta_yj)
                    xaj = (delta_xj / delta_yj) * (yaj - yoj) + xoj

                    ybj = (yfi * delta_yi * delta_yj + yfj * delta_xi * delta_xj + delta_yj * delta_xi * (xfi - xfj)) / \
                          (delta_xi * delta_xj + delta_yi * delta_yj)
                    xbj = (delta_xj / delta_yj) * (ybj - yfj) + xfj

                x = [xoj, xfj, xaj, xbj]
                y = [yoj, yfj, yaj, ybj]
                x.sort()
                y.sort()
                xcj = x[1]
                ycj = y[1]
                xdj = x[2]
                ydj = y[2]

                pj = math.sqrt((ydj - ycj)**2 + (xdj - xcj)**2)
                lj = math.sqrt((yfj - yoj)**2 + (xfj - xoj)**2)

                """IMPORTED CODE"""
                polygon = Polygon([(xoj, yoj), (xoj, yfj), (xfj, yfj), (xfj, yoj)])
                c_point = Point(xcj, ycj)
                d_point = Point(xdj, ydj)

                if (polygon.contains(c_point) or polygon.touches(c_point)) and (
                        polygon.contains(d_point) or polygon.touches(d_point)):
                    overlap = True
                else:
                    overlap = False
                """END IMPORTED CODE"""

                # Check if segments overlap
                pij = pj / (lj if lj != 0 else 0.1) * (1 if overlap else -1)

                pij = abs(pij)
                dija = abs(pj)

                if pij < self.threshold_overlap or dija > self.threshold_paralldist:
                    continue

                # PART 3: Perpendicular distance check
                xmj = (xcj + xdj) / 2.0
                ymj = (ycj + ydj) / 2.0

                if delta_xi == 0:
                    deij = int(xmj) - int(xoi)
                elif delta_yi == 0:
                    deij = int(ymj) - int(yoi)
                else:
                    deij = ((xmj - xoi) - (ymj - yoi) * delta_xi / delta_yi) / \
                            ((delta_xi ** 2) / (delta_yi ** 2) + 1) ** 0.5

                deij = abs(deij)
                if deij > self.threshold_perpendist:
                    continue

                # Grouping lines
                if line.group is None and line2.group is None:
                    group_idx += 1
                    line.group = group_idx
                    line2.group = group_idx
                elif line.group is None:
                    line.group = line2.group
                elif line2.group is None:
                    line2.group = line.group
                elif line.group is not None and line2.group is not None and line.group != line2.group:
                    # Join groups together by re-assigning all of line2's group to line1's group
                    old_group = line2.group
                    for line3 in lines:
                        if line3.group == old_group:
                            line3.group = line.group

            # If still no group, and none closest belong to, assign a new one with just this element
            if line.group is None:
                group_idx += 1
                line.group = group_idx

        # Display line groups
        if self.verbose:
            import cv2
            image_copy = image.copy()
            for line in lines:
                blue = (line.group * 100) % 255
                green = (line.group * 200) % 255
                red = (line.group * 300) % 255

                cv2.line(image_copy, (line.start.x, line.start.y), (line.end.x, line.end.y),
                         (blue, green, red), 3)

            from .dimension import Dimension
            max_dimension = Dimension(800, 800)
            display_dimension = Dimension(image_copy.shape[1], image_copy.shape[0])
            display_dimension.fitInside(max_dimension)
            image_copy = cv2.resize(image_copy, tuple(display_dimension))

            cv2.imshow("Debug Image", image_copy)
            cv2.waitKey(0)

        if self.verbose:
            print("Total Groups Found: ", group_idx)

        bbox_table = [[-1, -1, -1, -1]] * group_idx
        for line in lines:
            bbox_table[line.group - 1] = [
                min(line.start.x, line.end.x) if bbox_table[line.group - 1][0] == -1 else
                min(bbox_table[line.group - 1][0], min(line.start.x, line.end.x)),
                max(line.start.x, line.end.x) if bbox_table[line.group - 1][1] == -1 else
                max(bbox_table[line.group - 1][1], max(line.start.x, line.end.x)),
                min(line.start.y, line.end.y) if bbox_table[line.group - 1][2] == -1 else
                min(bbox_table[line.group - 1][2], min(line.start.y, line.end.y)),
                max(line.start.y, line.end.y) if bbox_table[line.group - 1][3] == -1 else
                max(bbox_table[line.group - 1][3], max(line.start.y, line.end.y)),
            ]

        # Add 5 pixels of padding to all boxes
        padding = 5
        for i in range(len(bbox_table)):
            if bbox_table[i] is not None:
                bbox_table[i] = [
                    max(0, bbox_table[i][0] - padding),
                    min(image.shape[1], bbox_table[i][1] + padding),
                    max(0, bbox_table[i][2] - padding),
                    min(image.shape[0], bbox_table[i][3] + padding),
                ]

        return coco_to_layout(bbox_to_coco(bbox_table))

    def image_loader(self, image: Union["np.ndarray", "Image.Image"]):
        # Convert PIL Image Input
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = np.array(image)

        return image


def bbox_to_coco(bboxes):
    images = []
    annotations = []
    categories = [{
        "id": 1,
        "name": "Paragraph",
        "supercategory": None
    }]

    ann_id = 1
    for bbox in bboxes:
        x1, x2, y1, y2 = bbox
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1

        annotation = {
            "id": ann_id,
            "image_id": 1,
            "category_id": 1,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
            # polygon segmentation is optional; empty list allowed
            "segmentation": [],
        }
        annotations.append(annotation)
        ann_id += 1

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    return coco_json


def coco_to_layout(coco_json):
    layout = Layout()

    for ele in coco_json["annotations"]:

        x, y, w, h = ele['bbox']

        layout.append(
            TextBlock(
                block=Rectangle(x, y, w + x, h + y),
                type="Paragraph",
                id=ele['id']
            )
        )

    return layout
