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

DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}


# inspired by https://pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
# rather than doing some fancy model we can instead iterate a segment over our segment display
# if the segment is 'on' then record it, otherwise don't
# use a predefined lookup table to work out which on segments correspond to which digit
def get_digit_num(img_path):
    # make image b/w
    b_w = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # threshold it
    digit = cv2.threshold(b_w, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # number 1 digit will only have 2 segments - computation will fail
    # just count number of countours and if we only have two its probs a 1
    num_cnts, _ = cv2.findContours(digit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(num_cnts) == 2:
        print(1)
        return

    # precomputed width and height of segments
    h, w = digit.shape
    dW, dH = (int(w * 0.25), int(h * 0.15))
    dHC = int(h * 0.05)

    segments = [
        ((0, 0), (w, dH)),  # top
        ((0, 0), (dW, h // 2)),  # top-left
        ((w - dW, 0), (w, h // 2)),  # top-right
        ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
        ((0, h // 2), (dW, h)),  # bottom-left
        ((w - dW, h // 2), (w, h)),  # bottom-right
        ((0, h - dH), (w, h)),  # bottom
    ]
    on = [0] * len(segments)

    for i, ((xA, yA), (xB, yB)) in enumerate(segments):
        segROI = digit[yA:yB, xA:xB]
        total = cv2.countNonZero(segROI)
        area = (xB - xA) * (yB - yA)
        if total / float(area) > 0.5:
            on[i] = 1

    digit = DIGITS_LOOKUP[tuple(on)]
    print(digit)


def get_nearest_therm_digit(img):
    pass


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
    get_digit_num(image_path)
    output_path = f"output/task3/result.txt"
    save_output(output_path, "Task 3 output", output_type="txt")


if __name__ == "__main__":
    run_task3("data/task3/lcd2/d1.png", None)
