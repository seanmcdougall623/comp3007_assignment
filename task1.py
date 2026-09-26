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

from lib.utils import order_corners


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


def identify_object(img_path):
    img = cv2.imread(img_path)

    model = YOLO("models/bp_therm_ident.pt")

    results = model.predict(img)

    # output has shape (img_cls, conf, (xywhr))
    output = []

    for result in results:
        for box in result.obb:
            corners = box.xyxyxyxy
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if conf > 0.7:
                output.append((class_name, round(conf, 3), corners))

    return output


def extract_item(img, bb, item_type):
    # load in image
    img = cv2.imread(img)

    # convert to np and order in l->r, t->b
    corners = order_corners(
        np.array(bb.cpu().numpy(), dtype=np.float32).reshape(4, 2)
    )

    tl, tr, br, bl = corners
    # implementation inspired by https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    # use linalg normalisation cuz thats less words
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    max_width = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    max_height = max(int(heightA), int(heightB))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    # rotate if we aren't as expected
    if warped.shape[1] > warped.shape[0] and item_type == "therm":
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # after skewing, image can sometimes not be perfectly flat
    # use hough transformation to ensure flat as possible

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    h, w = gray.shape

    threshold = min(w, h) // 4

    lines = cv2.HoughLines(edges, 1, np.pi / 360, threshold=threshold)

    # find LCD screen lines if bp, otherwise grab a few mercury lines
    max_lines = 4 if item_type == "bp" else 6

    # clean up this code cause it kinda aaa
    angles = []
    for _, theta in lines[:max_lines, 0]:
        # find angle of lines
        # maximum orientation is gonna be 45 degrees otherwise it will wrap around
        # genuinely do not know why but only seems to work in degs for some reason think my math is bad
        orient = (theta * 180.0 / np.pi) - 90.0
        orient = ((orient + 45) % 90) - 45
        if abs(orient) <= 45:
            angles.append(orient)

    correction = float(np.median(angles))
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, correction, 1.0)

    out = cv2.warpAffine(
        warped,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # cv2.imshow("Cropped BP", out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite(f"output/task1/{item_type}_cropped.png", out)


def run_task1(image_path, config):
    out = identify_object(image_path)
    for item in out:
        # TODO: decide whether to just use one function for both cause it seems pretty stable
        if not item:
            print("No valid object detected in the image.")
            return
        extract_item(image_path, item[2], item[0])

    # output_path = f"output/task1/result.txt"
    # save_output(output_path, "Task 1 output", output_type="txt")


if __name__ == "__main__":
    run_task1("./data/task1/img9.jpg", None)
