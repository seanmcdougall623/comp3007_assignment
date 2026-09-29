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
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from lib.utils import correct_skew, order_corners


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


def identify_object(img_path, loaded=False):
    if not loaded:
        img = cv2.imread(img_path)
    else:
        img = img_path

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

    bb = np.array(bb.cpu().numpy(), dtype=np.float32).reshape(4, 2)

    bb = order_corners(bb)

    warped, _ = correct_skew(img, bb)

    # after skewing, image can sometimes not be perfectly flat
    # use hough transformation to ensure flat as possible

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    h, w = gray.shape

    threshold = min(w, h) // 4

    lines = cv2.HoughLines(edges, 1, np.pi / 360, threshold=threshold)

    # find LCD screen lines if bp, otherwise grab a few mercury lines
    max_lines = 4 if item_type == "bp" else 8

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

    h, w = out.shape[:2]

    # rotate if we aren't as expected
    # short side always at bottom
    if out.shape[1] > out.shape[0]:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return out


def run_task1(image_path, config):
    inPath = Path(image_path)
    # iterate through folder

    for file in inPath.iterdir():
        if file.is_file() and file.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
        ]:
            out = identify_object(str(file))
            if not out:
                print(f"No valid object detected in {file}.")
                continue

            num = file.stem.split("img")[-1]
            item = out[0]  # select first item as per task sheet
            out_img = extract_item(str(file), item[2], item[0])
            f_name = ("lcd" if item[0] == "bp" else "therm") + num
            output_path = f"output/task1/{f_name}.jpg"
            save_output(output_path, out_img, output_type="image")
        else:
            print(f"Skipping {file.name}, not a valid image file.")


# testing code
if __name__ == "__main__":
    run_task1("./data/task1", None)
