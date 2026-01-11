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


"""
    def process_tmp(self, page: Page):
        image = page.image.copy()

        # Get lines and remove dots
        _my_lines = page.lines
        my_lines = []
        for _my_line in _my_lines:
            if _my_line.start.x - _my_line.end.x == 0 and _my_line.start.y - _my_line.end.y == 0:
                continue
            else:
                my_lines.append(_my_line)

        # Sorting lines
        my_lines.sort(key=lambda line: ((line.start.y + line.end.y) / 2, (line.start.x + line.end.x) / 2))

        my_lines_in_group = []
        # fix: include all indices (no off-by-one)
        my_lines_no_group = list(range(len(my_lines)))

        max_loop = len(my_lines)
        for act_loop in range(max_loop):
            if (len(my_lines_in_group) == 0) and (len(my_lines_no_group) == 0):
                break

            i = -1
            if self.EARLY_SKIP:
                early_skip = threshold_early_skip

            if len(my_lines_in_group) == 0:
                for candidate_line_idx in my_lines_no_group[:]:
                    x_o_i = my_lines[candidate_line_idx].start.x
                    y_o_i = image.shape[0] - my_lines[candidate_line_idx].start.y
                    x_f_i = my_lines[candidate_line_idx].end.x
                    y_f_i = image.shape[0] - my_lines[candidate_line_idx].end.y
                    delta_x_i = float(x_f_i - x_o_i)
                    delta_y_i = float(y_f_i - y_o_i)
                    if delta_x_i != 0 and delta_y_i != 0:  # Found!
                        i = candidate_line_idx
                        my_lines_no_group.remove(candidate_line_idx)
                        break
            else:
                i = my_lines_in_group.pop(0)

            if i == -1:
                break

            if len(my_lines_no_group) == 0:
                break
            else:
                my_lines[i].noise = False

                for j in my_lines_no_group[:]:
                    if self.EARLY_SKIP and early_skip < 0:
                        break

                    same_group = False

                    # Point setting
                    x_o_i = my_lines[i].start.x
                    y_o_i = image.shape[0] - my_lines[i].start.y
                    x_f_i = my_lines[i].end.x
                    y_f_i = image.shape[0] - my_lines[i].end.y

                    x_o_j = my_lines[j].start.x
                    y_o_j = image.shape[0] - my_lines[j].start.y
                    x_f_j = my_lines[j].end.x
                    y_f_j = image.shape[0] - my_lines[j].end.y

                    delta_x_i = float(x_f_i - x_o_i)
                    delta_y_i = float(y_f_i - y_o_i)
                    delta_x_j = float(x_f_j - x_o_j)
                    delta_y_j = float(y_f_j - y_o_j)

                    # skip dot lines
                    if delta_x_j == 0 and delta_y_j == 0:
                        my_lines_no_group.remove(j)
                        continue

                    # fix: correct angle computation (difference of angles)
                    theta_i = math.atan2(delta_y_i, delta_x_i)
                    theta_j = math.atan2(delta_y_j, delta_x_j)
                    theta_i_j = abs(theta_j - theta_i)
                    # normalize to [0, pi]
                    if theta_i_j > math.pi:
                        theta_i_j = 2 * math.pi - theta_i_j

                    x_a_j = (x_o_i * delta_x_i * delta_x_j + x_o_j * delta_y_i * delta_y_j + delta_x_j * delta_y_i * (
                            y_o_i - y_o_j)) / (delta_y_i * delta_y_j + delta_x_i * delta_x_j + self.EPS)
                    if delta_x_j != 0:
                        y_a_j = (delta_y_j / delta_x_j) * (x_a_j - x_o_j) + y_o_j
                    else:
                        x_a_j = y_o_j

                    x_b_j = (x_f_i * delta_x_i * delta_x_j + x_f_j * delta_y_i * delta_y_j + delta_x_j * delta_y_i * (
                            y_f_i - y_f_j)) / (delta_y_i * delta_y_j + delta_x_i * delta_x_j + self.EPS)
                    if delta_x_j != 0:
                        y_b_j = (delta_y_j / delta_x_j) * (x_b_j - x_f_j) + y_f_j
                    else:
                        x_b_j = y_f_j

                    c_d_candidates = [(x_o_j, y_o_j), (x_f_j, y_f_j), (x_a_j, y_a_j), (x_b_j, y_b_j)]
                    if delta_x_j != 0:
                        c_d_candidates.sort(key=lambda x: x[0])  # sort by x
                    elif delta_y_j != 0:
                        c_d_candidates.sort(key=lambda x: x[1])  # sort by y
                    x_c_j, y_c_j = c_d_candidates[1]
                    x_d_j, y_d_j = c_d_candidates[2]

                    polygon = Polygon([(x_o_j, y_o_j), (x_o_j, y_f_j), (x_f_j, y_f_j), (x_f_j, y_o_j)])
                    c_point = Point(x_c_j, y_c_j)
                    d_point = Point(x_d_j, y_d_j)

                    if (polygon.contains(c_point) or polygon.touches(c_point)) and (
                            polygon.contains(d_point) or polygon.touches(d_point)):
                        overlap = True
                    else:
                        overlap = False

                    p_j = math.sqrt(math.pow(y_d_j - y_c_j, 2) + math.pow(x_d_j - x_c_j, 2))
                    l_j = math.sqrt(math.pow(y_f_j - y_o_j, 2) + math.pow(x_f_j - x_o_j, 2))
                    if l_j == 0:
                        l_j = 0.1
                    if overlap:
                        p_i_j = p_j / l_j
                    else:
                        p_i_j = -p_j / l_j

                    # Calculate parallel_dist
                    if overlap:
                        d_i_j_a = p_j
                    else:
                        d_i_j_a = -p_j

                    # Calculate perpend_dist
                    x_m_j = (x_c_j + x_d_j) / 2.0
                    y_m_j = (y_c_j + y_d_j) / 2.0

                    if delta_x_i != 0.0 and delta_y_i != 0.0:
                        d_e_i_j = ((x_m_j - x_o_i) - (y_m_j - y_o_i) * delta_x_i / (delta_y_i + self.EPS)) / (
                                (delta_x_i ** 2) / (delta_y_i ** 2 + self.EPS) + 1) ** 0.5
                    elif delta_y_i == 0.0:
                        d_e_i_j = int(y_m_j) - int(y_o_i)
                    elif delta_x_i == 0.0:
                        d_e_i_j = int(x_m_j) - int(x_o_i)
                    d_e_i_j = abs(d_e_i_j)

                    # DECIDE GROUPNESS (same logic)
                    if theta_i_j < threshold_angle:
                        if 0 < d_e_i_j < threshold_perpendist:
                            if overlap and p_i_j <= threshold_overlap:
                                same_group = True
                            elif abs(d_i_j_a) < threshold_paralldist:
                                same_group = True

                    if same_group:
                        # assign group to i and j and enqueue j for further grouping
                        if my_lines[i].group is None:
                            group_idx += 1
                            my_lines[i].group = group_idx
                        # assign j to the same group and remove from ungrouped list
                        my_lines[j].group = my_lines[i].group
                        if j in my_lines_no_group:
                            my_lines_no_group.remove(j)
                        my_lines_in_group.append(j)

                        if self.EARLY_SKIP:
                            early_skip = threshold_early_skip
                    else:
                        if self.EARLY_SKIP:
                            early_skip -= 1

                # after checking js, if i still has no group assign a new one
                if (not my_lines[i].noise) and (my_lines[i].group is None):
                    group_idx += 1
                    my_lines[i].group = group_idx

        dist = [10, 20, 20, 50, 20, 20, 30, 25, 25, 25, 25, 25, 25]
        n, bins, patches = plt.hist(dist, np.max(dist) - np.min(dist) + 1, facecolor='orange', alpha=0.5)
        plt.close()
        n_copy = n.copy()
        n_copy[::-1].sort()

        a = np.where(n == n_copy[2])

        if len(a[0]) > 1:
            _max = a[0][int(len(a[0]) / 2)]
        else:
            _max = a[0][0]

        _max + np.min(dist)

        y_o_i = 4875
        x_f_i = 2676
        y_f_i = 4872

        x_o_j = 2476
        y_o_j = 1938
        x_f_j = 2840
        y_f_j = 1932
        x_a_j = 2363
        y_a_j = 1939
        x_b_j = 2631
        y_b_j = 1935

        c_d_candidates = [(x_o_j, y_o_j), (x_f_j, y_f_j), (x_a_j, y_a_j), (x_b_j, y_b_j)]
        c_d_candidates.sort(key=lambda x: x[0])

        Polygon([(0 - 1, y_o_i + 1), (x_f_i + 1, y_f_i + 1), (x_f_j + 1, y_f_j - 1), (x_o_j - 1, y_o_j - 1)])

        image = page.image.copy()
        for my_line in my_lines:
            if my_line.group is None:
                continue
            else:
                blue = (my_line.group * 100) % 255
                green = (my_line.group * 200) % 255
                red = (my_line.group * 300) % 255

            cv2.line(image, (my_line.start.x, my_line.start.y), (my_line.end.x, my_line.end.y), (blue, green, red),
                     10)

        max_dimension = Dimension(image.shape[1], image.shape[0])
        display_dimension = Dimension(image.shape[1], image.shape[0])
        display_dimension.fitInside(max_dimension)

        image = page.image.copy()
        tot_groups = group_idx + 1
        group_table = []
        for group_idx in range(tot_groups):
            group_table.append([])

        for my_line in my_lines:
            for group_idx in range(1, tot_groups):
                if my_line.group is None:
                    continue
                elif my_line.group == group_idx:
                    exaggerated_left_start_x = my_line.start.x - self.THRESHOLD_POLY_EXAGGERATE
                    exaggerated_up_start_y = my_line.start.y + self.THRESHOLD_POLY_EXAGGERATE
                    exaggerated_down_start_y = my_line.start.y - self.THRESHOLD_POLY_EXAGGERATE

                    exaggerated_right_end_x = my_line.end.x + self.THRESHOLD_POLY_EXAGGERATE
                    exaggerated_up_end_y = my_line.end.y + self.THRESHOLD_POLY_EXAGGERATE
                    exaggerated_down_end_y = my_line.end.y - self.THRESHOLD_POLY_EXAGGERATE

                    group_table[group_idx - 1].append([exaggerated_left_start_x, exaggerated_up_start_y])
                    group_table[group_idx - 1].append([exaggerated_left_start_x, exaggerated_down_start_y])

                    group_table[group_idx - 1].append([exaggerated_right_end_x, exaggerated_up_end_y])
                    group_table[group_idx - 1].append([exaggerated_right_end_x, exaggerated_down_end_y])

        for group_idx in range(1, tot_groups):
            points = np.array(group_table[group_idx - 1], dtype='int')
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
            box = np.intp(box)
            cv2.drawContours(image, np.int32([box]), 0, (0, 0, 255), 7)

        image = page.image.copy()
        bbox_table = np.zeros((group_idx + 1, 4))  # [min_x,max_x,min_y,max_y]
        bbox_table[:, 0] = image.shape[1]
        bbox_table[:, 1] = 0
        bbox_table[:, 2] = image.shape[0]
        bbox_table[:, 3] = 0

        # Find BoundingBoxes for Each Group
        for my_line in my_lines:
            for i in range(1, group_idx + 1):
                if my_line.group is None:
                    # Update if found new min or max
                    if my_line.start.x < bbox_table[-1, 0]:
                        bbox_table[-1, 0] = my_line.start.x
                    if my_line.end.x > bbox_table[-1, 1]:
                        bbox_table[-1, 1] = my_line.end.x
                    if my_line.start.y < bbox_table[-1, 2]:
                        bbox_table[-1, 2] = my_line.start.y
                    if my_line.end.y > bbox_table[-1, 3]:
                        bbox_table[-1, 3] = my_line.end.y
                elif my_line.group == i:
                    # Update if found new min or max
                    if my_line.start.x < bbox_table[i - 1, 0]:
                        bbox_table[i - 1, 0] = my_line.start.x
                    if my_line.end.x > bbox_table[i - 1, 1]:
                        bbox_table[i - 1, 1] = my_line.end.x
                    if my_line.start.y < bbox_table[i - 1, 2]:
                        bbox_table[i - 1, 2] = my_line.start.y
                    if my_line.end.y > bbox_table[i - 1, 3]:
                        bbox_table[i - 1, 3] = my_line.end.y

    def process_tmp_2(self, page: Page):
        # Get lines and remove dots
        _lines = page.lines
        lines = []
        for _line in _lines:
            if _line.start.x - _line.end.x == 0 and _line.start.y - _line.end.y == 0:
                continue
            else:
                lines.append(_line)

        # Sorting lines
        lines.sort(key=lambda line: ((line.start.y + line.end.y) / 2, (line.start.x + line.end.x) / 2))

        lines_in_group = []
        lines_no_group = lines[:]

        group_idx = 0
        numer_of_lines = len(lines)
        for loop_idx in range(numer_of_lines):
            if (len(lines_in_group) == 0) and (len(lines_no_group) == 0):
                raise Exception("No lines found!")  # TODO: preveri če je to res

            # There are no more ungrouped lines
            if len(lines_no_group) == 0:
                break

            line_candidate = lines_in_group[0]
            lines_no_group.remove(0)

            # Ignore noisy lines
            if line_candidate.noise:
                continue

            for line_in_group_candidate in lines_in_group:
                same_group = False

                # Point setting
                x_o_i = line_in_group_candidate.start.x
                y_o_i = image.shape[0] - line_in_group_candidate.start.y
                x_f_i = line_in_group_candidate.end.x
                y_f_i = image.shape[0] - line_in_group_candidate.end.y

                x_o_j = line_candidate.start.x
                y_o_j = image.shape[0] - line_candidate.start.y
                x_f_j = line_candidate.end.x
                y_f_j = image.shape[0] - line_candidate.end.y

                delta_x_i = float(x_f_i - x_o_i)
                delta_y_i = float(y_f_i - y_o_i)
                delta_x_j = float(x_f_j - x_o_j)
                delta_y_j = float(y_f_j - y_o_j)

                # fix: correct angle computation (difference of angles)
                theta_i = math.atan2(delta_y_i, delta_x_i)
                theta_j = math.atan2(delta_y_j, delta_x_j)
                theta_i_j = abs(theta_j - theta_i)
                # normalize to [0, pi]
                if theta_i_j > math.pi:
                    theta_i_j = 2 * math.pi - theta_i_j

                # DECIDE GROUPNESS (same logic)
                if theta_i_j < self.threshold_angle:
                    # Further checks go here (perpend_dist, overlap, paralldist)
                    same_group = True  # Placeholder for further checks

                if same_group:
                    # assign group to i and j and enqueue j for further grouping
                    if line_in_group_candidate.group is None:
                        group_idx += 1
                        line_in_group_candidate.group = group_idx
                    # assign j to the same group and remove from ungrouped list
                    line_candidate.group = line_in_group_candidate.group
                    if line_candidate in lines_no_group:
                        lines_no_group.remove(line_candidate)
                    lines_in_group.append(line_candidate)

            # After all checks, if still no group, assign a new one
            if line_candidate.group is None:
                group_idx += 1
                line_candidate.group = group_idx

        return None
"""