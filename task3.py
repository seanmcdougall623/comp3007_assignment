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

from lib.svm import test_bp_digit, test_therm_digit
from lib.utils import extract_digits, extract_red


def get_digit_num(img_path):
    return test_bp_digit(img_path)


# template match to find the nearest digits
# @param img must be grayscale
# @returns list of 2 digits (val, x, y, w, h)
def get_nearest_therm_digits(img):
    b_w = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # crop down to left hand third of image so we only get the digits in celsius
    third_img = b_w[:, : b_w.shape[1] // 3]

    # copy and gaussian blur to reduce noise
    filtered = cv2.GaussianBlur(third_img, (5, 5), 0)
    h, w = filtered.shape
    ref_cnt = extract_digits(
        filtered, kernel_size=(1, 1), area_threshold=h * w // 100
    )

    # can sometimes falsely include the shadow from the thermometer
    # just remove it
    if len(ref_cnt) > 4:
        ref_cnt = ref_cnt[1:5]

    # we'll have 4 contours, in order from top->bottom, l->r
    # take 1 and 3 as thats the digits we are interested in - other 2 are gonna be 0s and classifier doesn't need em
    digits = [ref_cnt[0], ref_cnt[2]]
    output = []

    # classify them
    for digit in digits:
        x, y, w, h = digit
        d_img = third_img[y : y + h, x : x + w]
        num = test_therm_digit(d_img, loaded=True)
        # sanity check in case misclassification
        # if we did misclassify just take the first digit as ground truth bcz it seems to be more acccurate in testing
        if len(output) >= 1:
            next_num = output[0][0] - 1
            if next_num != num:
                print(
                    "Classifying error when establishing second digit. Taking first as ground truth"
                )
                num = next_num

        output.append((num, x, y, w, h))

    return output


# take top and bottom value, then ranging will be fixed
# mercury value will just be fixed spacing and we can calculate from there
def get_therm_reading(img):
    img = cv2.imread(img)
    _, merc_y, _, _ = extract_red(img)
    digits = get_nearest_therm_digits(img)

    # don't need x values just y value and digit value
    (d1, _, y1, _, _) = digits[0]
    (d2, _, y2, _, h2) = digits[1]

    # sanity check in case misclassification
    if d1 < d2 or abs(d1 - d2) != 1:
        raise ValueError(
            f"Classification error! Received readings of {d1} and {d2}"
        )

    # calculate px spacing
    # 1u = 0.5 degrees
    px_spacing = abs(y1 - y2) // 10

    # calculate mercury reading
    # offset by midpoint of first reading (where the closest known value starts)
    merc_reading = ((y2 + h2 // 2) - merc_y) / px_spacing

    # add in temp offset

    merc_reading += d2 * 10
    # round to 1 decimal place cause we ain't that accurate
    return round(merc_reading, 1)


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
    for i in range(1, 8):
        print(get_digit_num(f"data/task3/lcd5/d{i}.png"))
    # merc_reading = get_therm_reading(image_path)
    # print(merc_reading)
    output_path = f"output/task3/result.txt"
    save_output(output_path, "Task 3 output", output_type="txt")


if __name__ == "__main__":
    run_task3("output/task2/t.png", None)
