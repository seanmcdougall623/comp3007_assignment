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
from ultralytics import YOLO

from lib.utils import get_digit_num_manual


def get_digit_num(img_path):
    # load image and threshold to b/w as thats what model was trained on
    b_w = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    digit = cv2.threshold(b_w, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # tiny filtering step
    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.morphologyEx(digit, cv2.MORPH_OPEN, kernel)

    # pad out any dimension < 100 to 100 so it doesnt do any goofy stretching
    h, w = mask.shape
    if h < 128 or w < 128:
        pad_h = max(0, 128 - h)
        pad_w = max(0, 128 - w)
        mask = cv2.copyMakeBorder(
            mask, pad_h, 0, pad_w, 0, cv2.BORDER_CONSTANT, value=255
        )

    img_fin = cv2.resize(mask, (128, 128), cv2.INTER_AREA)

    model = YOLO("./models/digit_cls.pt")

    result = model.predict(img_fin)[0]

    confidence = result.probs.top1conf
    if confidence < 0.8:
        # method expects inverted
        digit = cv2.threshold(
            b_w, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]
        return get_digit_num_manual(digit)
    else:
        return int(result.probs.top1)


# template match to find the nearest digits
# @param img must be grayscale
def get_nearest_therm_digits(img):
    b_w = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base_path = str(os.getcwd()) + "/lib/therm_digits/"
    # stupid lazy way to iterate through each template
    for i in range(-2, 6):
        template = cv2.imread(base_path + f"{i}.png", cv2.IMREAD_GRAYSCALE)
        h, w = template.shape

        # see if we have the value in our image
        # confidence needs to be >80%
        res = cv2.matchTemplate(b_w, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print(f"{i}: Max val: {max_val} @ {max_loc}")
        bottom_right = (max_loc[0] + w, max_loc[1] + h)
        cv2.rectangle(img, max_loc, bottom_right, i + 120, 2)

    cv2.imshow("output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# take top and bottom value, then ranging will be fixed
# mercury value will just be fixed spacing and we can calculate from there
def get_therm_reading(img):
    pass


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


def run_task3(image_path, config):
    # TODO: Implement task 3 here
    digit = get_digit_num(image_path)
    print(digit)
    output_path = f"output/task3/result.txt"
    save_output(output_path, "Task 3 output", output_type="txt")


if __name__ == "__main__":
    run_task3("data/task3/lcd4/d7.png", None)
