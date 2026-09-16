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
            ref_cnt.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]))
    if draw:
        cv2.drawContours(img, ref_cnt, -1, (0, 255, 0), 2)

    return ref_cnt, img


def export_lcd_digits(img, cnt, save_dir):
    for i, digit in enumerate(cnt):
        # lazy way to find the x/y coords (im tired)
        x, y, w, h = cv2.boundingRect(digit)
        cropped_img = img[y : y + h, x : x + w]
        cv2.imwrite(save_dir + f"d{i+1}.png", cropped_img)


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
    cnt, img = find_lcd_digits(image_path)
    export_lcd_digits(img, cnt, "output/task2/")
    # output_path = "output/task2/result.txt"
    # save_output(output_path, "Task 2 output", output_type="txt")


if __name__ == "__main__":
    run_task2("./data/task2/lcd5.png", None)
