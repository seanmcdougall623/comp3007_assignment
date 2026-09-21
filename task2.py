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


# Author: [Your Name]
# Last Modified: 2024-09-09

import os

import cv2
import numpy as np
import math

from lib.utils import preprocess_image


def find_lcd_digits(img, draw=False):
    # processings
    img = cv2.imread(img)
    pre_img = preprocess_image(img, resize=False)

    # thresholding
    img_edges = cv2.Canny(pre_img, threshold1=5, threshold2=120)

    # dilate the threshold a little to connect the segments
    kernel = np.ones((7, 7), np.uint8)
    img_dil = cv2.dilate(img_edges, kernel, iterations=1)

    contours, _ = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ref_cnt = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # check if sufficiently big, and rectangular shape
        if area > 2000 and (w * 1.3 < h):
            # if we want to draw use direct contour approximation
            if draw:
                ref_cnt.append(
                    np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                )
            else:
                # much more computational efficent
                ref_cnt.append((x, y, w, h))

    # sort based on pixel position
    # take into account both x and y
    # idk probably not efficient but shld be fine
    ref_cnt = sorted(ref_cnt, key=lambda r: math.sqrt(r[0] ** 2 + r[1] ** 2))

    # only draw if we wanna see the output
    if draw:
        cv2.drawContours(img, ref_cnt, -1, (0, 255, 0), 2)

    return ref_cnt, img


def export_lcd_digits(img, cnt, save_dir):
    for i, digit in enumerate(cnt):
        x, y, w, h = digit
        cropped_img = img[y : y + h, x : x + w]
        cv2.imwrite(save_dir + f"d{i+1}.png", cropped_img)


def find_thermo_section(img):
    img = cv2.imread(img)

    img_h, img_w, _ = img.shape

    # approach is to look for red line as thats only thing consistent
    # line is too skinny to do any line/edge detection

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # !!!
    # TODO: REFERENCE: AI GENERATED
    # !!!
    lower = np.array([5, 55, 90])
    upper = np.array([25, 255, 200])
    # ai stuff ends

    red_mask = cv2.inRange(img_hsv, lower, upper)
    # use morphological filter to filter out noise
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # work out where largest blob is and grab it
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # grab coords of where the box is
    x, y, w, h, _ = stats[largest]

    # now that we know the x/y coords of where the mercury goes to, we just need to find our closest 'big' lines

    # use hough transform to find da lines

    top_point = (x + w // 2, y + int(img_h * 0.015))
    half_height = int(img_h * 0.06)
    y0 = max(0, top_point[1] - half_height)
    y1 = min(img_h, top_point[1] + half_height)
    box = (0, y0, img_w, y1 - y0)
    return box, img


def extract_thermo_section(img, box, dir):
    x, y, w, h = box
    cropped_img = img[y : y + h, x : x + w]
    cv2.imwrite(dir + "t.png", cropped_img)


def save_output(output_path, content, output_type="txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_type == "txt":
        with open(output_path, "w") as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == "image":
        # Assuming 'content' is a valid image object, e.g., from OpenCV
        cv2.imwrite(output_path, content)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")


def run_task2(image_path, config):
    # TODO: Implement task 2 here
    box, img = find_thermo_section(image_path)
    extract_thermo_section(img, box, "output/task2/")
    # cv2.imshow("output", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # output_path = "output/task2/result.txt"
    # save_output(output_path, "Task 2 output", output_type="txt")


if __name__ == "__main__":
    run_task2("./data/task2/thermo1.png", None)
