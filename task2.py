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

from lib.utils import extract_digits, extract_red, preprocess_image


def find_lcd_digits(img, draw=False):
    # processings
    img = cv2.imread(img)
    pre_img = preprocess_image(img, resize=False)

    ref_cnt = extract_digits(
        pre_img, kernel_size=(7, 7), area_threshold=1000, draw=draw
    )

    # only draw if we wanna see the output
    if draw:
        cv2.drawContours(img, ref_cnt, -1, (0, 255, 0), 2)
        cv2.imshow("Recognised Digits", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

    # cut down to middle tube, and remove bottom 20% (bulb)
    tube_w = img_w * 0.1
    cx = img_w // 2

    xl = int(cx - tube_w // 2)
    xr = int(cx + tube_w // 2)

    tube_img = img[: int(img_h * 0.8), xl:xr]

    x, y, w, _ = extract_red(tube_img)

    y0 = int(max(0, y - img_h * 0.06))
    y1 = int(min(img_h, y + img_h * 0.08))
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
    # ref_cnt, img = find_lcd_digits(image_path, draw=False)
    # export_lcd_digits(img, ref_cnt, "output/task2/")
    box, img = find_thermo_section(image_path)
    extract_thermo_section(img, box, "output/task2/")
    # cv2.imshow("output", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # output_path = "output/task2/result.txt"
    # save_output(output_path, "Task 2 output", output_type="txt")


if __name__ == "__main__":
    run_task2("./data/task2/thermo3.png", None)
