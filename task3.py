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

from lib.svm import test_digit


def get_digit_num(img_path):
    return test_digit(img_path)


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
