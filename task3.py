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

import math
import os
from pathlib import Path

import cv2

from lib.svm import test_bp_digit, test_therm_digit
from lib.utils import check_negative, extract_red, extract_therm_digits


def get_digit_num(img_path, loaded=False):
    return test_bp_digit(img_path, loaded)


# template match to find the nearest digits
# @param img must be grayscale
# @returns list of 2 digits (val, x, y, w, h)
def get_nearest_therm_digits(img):

    ref_cnt, third_img = extract_therm_digits(
        img, area_threshold_scale=50, draw=False
    )

    # check to see if any numbers are negative (small horizontal feature)
    neg = check_negative(third_img, draw=False)

    # can sometimes falsely include the shadow from the thermometer
    # just remove it
    if len(ref_cnt) > 4:
        # only keep contours that are in the right side of the image, as the shadow is on the left
        for cnt in ref_cnt:
            if cnt[0] <= 0.25 * third_img.shape[1]:
                ref_cnt.remove(cnt)
                break

    # sometimes we can accidentally crop out our 0s when taking the left third
    if len(ref_cnt) == 2:
        digits = [ref_cnt[0], ref_cnt[1]]
    # tricky edge case - which is the 0 and which is the digit?
    # in this case pass through all values, and remove any 0s
    elif len(ref_cnt) == 3:
        digits = ref_cnt
    elif len(ref_cnt) == 4:
        # we'll have 4 contours, in order from top->bottom, l->r
        # take 1 and 3 as thats the digits we are interested in - other 2 are gonna be 0s and classifier doesn't need em
        digits = [ref_cnt[0], ref_cnt[2]]

    output = []
    # classify them
    for digit in digits:
        x, y, w, h = digit
        d_img = third_img[y : y + h, x : x + w]
        num = test_therm_digit(d_img, loaded=True)
        output.append([num, x, y, w, h])

    # remove any 0s if we just added
    # ofc which is the right 0 (what if the thermometer reads 0?)
    # then 0 will be in between
    if len(output) == 3:
        if output[1][0] == 0:
            output.remove(output[1])
        elif output[2][0] == 0:
            output.remove(output[2])

    # sanity check in case misclassification
    # if we did misclassify just take the highest digit as ground truth bcz it seems to be more acccurate in testing
    if len(output) == 2:
        d1 = output[0][0]
        d2 = output[1][0]
        if abs(d1 - d2) != 1 or d1 < d2:
            # use closest positive digit to 0 as probs the correct one
            truth = min(x for x in [d1, d2] if x > 0)
            print(
                f"Misclassficiation detected! Received {d1} and {d2}. Using {truth} as ground truth"
            )
            i = (d1, d2).index(truth)
            if i:
                output[0][0] = truth + 1
            else:
                output[1][0] = truth - 1

    # if we have a negative sign, fix digits
    if neg:
        for digit in output:
            if digit[0] > 0:
                digit[0] = -digit[0]

    return output


# take top and bottom value, then ranging will be fixed
# mercury value will just be fixed spacing and we can calculate from there
def get_therm_reading(img, loaded=False):
    if not loaded:
        img = cv2.imread(img)
    digits = get_nearest_therm_digits(img)

    # don't need x values just y value and digit value
    (d1, _, y1, _, _) = digits[0]
    (d2, _, y2, _, h2) = digits[1]

    # sanity check in case misclassification
    if d1 < d2 or abs(d1 - d2) != 1:
        raise ValueError(
            f"Classification error! Received readings of {d1} and {d2}"
        )
    else:
        print(f"Received readings of {d1} and {d2}")
    # calculate px spacing
    # 1u = 0.5 degrees
    px_spacing = abs(y1 - y2) // 10

    # loop through in case we extract the mercury a bit optimistically and place the temperature too high
    temp_reading = 99
    start_reg = 130
    while d1 - (temp_reading / 10) < 0 or (temp_reading / 10) - d2 > 1:
        _, merc_y, _, _ = extract_red(img, a_filt=start_reg)

        # calculate mercury reading
        # offset by midpoint of first reading (where the closest known value starts)
        merc_reading = ((y2 + h2 // 2) - merc_y) / px_spacing
        # add in temp offset
        merc_reading += d2 * 10
        temp_reading = round(merc_reading, 1)

        start_reg += 1

    # round to lowest number and take 1 as seems to be consistently off by 1 in testing
    # unless it takes us below the floor

    floored = math.floor(temp_reading)

    return floored - 1 if floored > d1 * 10 else floored


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
    inPath = Path(image_path)

    for folder in inPath.iterdir():
        if folder.is_dir():
            if folder.name.startswith("lcd"):
                for file in folder.iterdir():
                    if file.is_file() and file.suffix in [
                        ".jpg",
                        "jpeg",
                        ".png",
                    ]:
                        digit = get_digit_num(file)
                        output_path = (
                            f"output/task3/{folder.name}/{file.stem}.txt"
                        )
                        save_output(output_path, digit, output_type="txt")
                    else:
                        print(
                            f"Skipping {file.name}, not a valid image file."
                        )
            elif folder.name.startswith("thermo"):
                for file in folder.iterdir():
                    if file.is_file() and file.suffix in [
                        ".jpg",
                        "jpeg",
                        ".png",
                    ]:
                        reading = get_therm_reading(file)
                        output_path = (
                            f"output/task3/{folder.name}/{file.stem}.txt"
                        )
                        save_output(output_path, reading, output_type="txt")
                    else:
                        print(
                            f"Skipping {file.name}, not a valid image file."
                        )
            else:
                print(f"Skipping {folder.name}, not a valid image folder.")
        else:
            print(f"Skipping {folder.name}, not a folder.")


# debug purposes
if __name__ == "__main__":
    # !! uncomment for BP !!
    img_path = "output/task2/"
    for img in sorted(Path(img_path).iterdir()):
        if img.name == "t.jpg":
            continue
        print(get_digit_num(img))

    # !! uncomment for therm !!
    # img_path = "output/task2/"
    # read = get_therm_reading(img_path + "t.jpg")
    # print(read)
