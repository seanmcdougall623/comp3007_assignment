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
from matplotlib.pyplot import gray
from networkx import edges
from sympy import sift


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


# convert and filter out noise
def preprocess_image(img):
    resized = cv2.resize(img, (600, 480), interpolation=cv2.INTER_LINEAR)
    grayed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayed, (5, 5), 0)

    return blur


def extract_thermo(img):

    img_edges = cv2.Canny(img, threshold1=5, threshold2=120)

    contours, _ = cv2.findContours(
        img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    ref_cnt = []

    for cnt in contours:
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        area = cv2.contourArea(approx)
        if area > 100:
            ref_cnt.append(approx)

    cv2.drawContours(img, ref_cnt, -1, (0, 255, 0), 2)
    return img


def extract_bp(img):
    pass


def run_task1(image_path, config):
    # TODO: Implement task 1 here
    img = cv2.imread(image_path)
    preprocess = preprocess_image(img)
    out = extract_thermo(preprocess)
    cv2.imshow("output", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# output_path = f"output/task1/result.txt"
# save_output(output_path, "Task 1 output", output_type="txt")


if __name__ == "__main__":
    run_task1("./data/task1/img1.jpg", None)
