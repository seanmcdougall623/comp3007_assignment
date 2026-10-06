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

from lib.utils import (
    extract_digits,
    extract_red,
    extract_therm_digits,
    preprocess_image,
)


def find_lcd_digits(img, draw=False, loaded=False):
    # processings
    if not loaded:
        img = cv2.imread(img)
    pre_img = preprocess_image(img, resize=False)

    h, w = pre_img.shape

    ref_cnt = extract_digits(
        pre_img,
        kernel_size=(7, 7),
        threshold1=5,
        threshold2=90,
        area_threshold=h * w // 500,
        draw=draw,
    )

    # only draw if we wanna see the output
    if draw:
        cv2.drawContours(img, ref_cnt, -1, (0, 255, 0), 2)
        cv2.imshow("Recognised Digits", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return ref_cnt, img


def extract_lcd_digits(img, cnt, out_dir=None):
    output = []
    for i, digit in enumerate(cnt):
        x, y, w, h = digit
        cropped_img = img[y : y + h, x : x + w]
        if out_dir:
            output_path = os.path.join(out_dir, f"d{i + 1}.jpg")
            cv2.imwrite(output_path, cropped_img)
        output.append(cropped_img)
    return output


def find_thermo_section(img, loaded=False):
    if not loaded:
        img = cv2.imread(img)
    else:
        img = img.copy()
    img_h, img_w, _ = img.shape

    # approach is to look for red line as thats only thing consistent
    # line is too skinny to do any line/edge detection

    # cut down to middle tube, and remove bottom 20% (bulb)
    tube_w = img_w * 0.1
    cx = img_w // 2

    xl = int(cx - tube_w // 2)
    xr = int(cx + tube_w // 2)
    y_bot = int(img_h * 0.8)

    tube_img = img[:y_bot, xl:xr]

    mx, my, _, _ = extract_red(tube_img)

    # find digits closest to the mercury reading, and use that to crop the image down to just the thermometer section
    ref_cnt, _ = extract_therm_digits(img)

    # find two boxes closest
    # need to convert x,y from tube_img to img
    x_mapped = mx + xl
    closest = sorted(
        ref_cnt,
        # use euclidean distance to find distance from bounding box
        key=lambda b: math.sqrt((x_mapped - b[0]) ** 2 + (my - b[1]) ** 2),
    )

    # tree needs to be traversed to find
    # 1. top_digit needs to be > merc_y reading
    # 2. bottom_digit needs to be < merc_y reading
    # 3. bottom_digit needs to be > 5% away from top_digit, otherwise we've grabbed the wrong digit

    # add 2% of padding just in case the mercury reading is close to a digit
    tops = [b for b in closest if b[1] < my * 0.98]
    bottoms = [b for b in closest if b[1] > my * 0.98]

    top_digit = None
    bottom_digit = None

    found = False
    for t in tops:
        for b in bottoms:
            if abs(b[1] - t[1]) > 0.05 * img_h:
                top_digit = t
                bottom_digit = b
                found = True
                break
        if found:
            break

    y0 = top_digit[1] - 10
    y1 = bottom_digit[1] + bottom_digit[3] + 10

    box = (0, y0, img_w, y1 - y0)
    return box, img


def extract_thermo_section(img, box, out_dir=None):
    x, y, w, h = box
    cropped_img = img[y : y + h, x : x + w]
    if out_dir:
        cv2.imwrite(os.path.join(out_dir, "t.jpg"), cropped_img)

    return cropped_img


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
    inPath = Path(image_path)

    for file in inPath.iterdir():
        if file.is_file() and file.suffix in [".jpg", "jpeg", ".png"]:
            # name is either gonna be lcd or thermo, so just check first 3 chars
            f_name = file.stem
            if f_name[:3] == "lcd":
                ref_cnt, img = find_lcd_digits(file, draw=False)
                out = extract_lcd_digits(img, ref_cnt)
                for i, digit_img in enumerate(out):
                    output_path = (
                        Path(__file__).parent
                        / f"output/task2/{f_name}/d{i + 1}.jpg"
                    )
                    save_output(output_path, digit_img, output_type="image")
            elif f_name[:3] == "the":
                box, img = find_thermo_section(file)
                out = extract_thermo_section(img, box)
                save_output(
                    Path(__file__).parent / f"output/task2/{f_name}/t.jpg",
                    out,
                    output_type="image",
                )

        else:
            print(f"Skipping {file.name}, not a valid image file.")


# debug purposes
if __name__ == "__main__":
    # !! uncomment for BP !!
    # ref_cnt, img = find_lcd_digits("output/task1/lcd5.jpg", draw=False)
    # extract_lcd_digits(img, ref_cnt, "output/task2/")

    # !! uncomment for therm !!
    # box, img = find_thermo_section("output/task1/therm1.jpg")
    # extract_thermo_section(img, box, "output/task2/")

    # !! uncomment for task 2 !!
    run_task2("data/task2", None)
