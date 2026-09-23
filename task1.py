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

from lib.utils import preprocess_image


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


def extract_thermo(img, bb):
    # load in image
    img = cv2.imread(img)

    # convert to np
    corners = np.array(bb, dtype=np.float32)
    # padding if needed (really just leftover from testing)
    pad = 0

    (cx, cy), (w, h), angle = cv2.minAreaRect(corners)
    w, h = w + 2 * pad, h + 2 * pad

    # crop out interested section
    # code taken from https://github.com/ultralytics/ultralytics/issues/9344#issuecomment-2022372776

    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC)

    # calc points for crop
    x1, y1 = round(cx - w / 2), round(cy - h / 2)
    x2, y2 = round(cx + w / 2), round(cy + h / 2)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, W), min(y2, H)

    crop = rotated[y1:y2, x1:x2]

    # rotate if we aren't as expected
    if crop.shape[1] > crop.shape[0]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imshow("Cropped Thermometer", crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_bp(img, bb):
    pass


def run_task1(image_path, config):
    out = identify_object(image_path)
    for item in out:
        if item[0] == "therm":
            extract_thermo(image_path, item[2])
        elif item[0] == "bp":
            extract_bp(image_path, item[2])
        else:
            print("No valid object detected in the image.")
            return
    # output_path = f"output/task1/result.txt"
    # save_output(output_path, "Task 1 output", output_type="txt")


if __name__ == "__main__":
    run_task1("./data/task1/img6.jpg", None)
