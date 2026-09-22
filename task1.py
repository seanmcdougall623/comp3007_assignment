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
    img = cv2.resize(img, (800, 600))

    model = YOLO("models/bp_therm_ident.pt")

    results = model.predict(img)

    for result in results:
        print(result.obb)
        for box in result.obb:
            corners = box.xyxyxyxy[0]
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if confidence > 0.3:
                # Draw bounding box
                points = np.array(corners.cpu(), dtype=np.int32).reshape(
                    (-1, 1, 2)
                )
                cv2.polylines(
                    img,
                    [points],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )

    cv2.imshow("Predictions", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_thermo(img):
    pass


def extract_bp(img):
    pass


def run_task1(image_path, config):
    identify_object(image_path)
    # output_path = f"output/task1/result.txt"
    # save_output(output_path, "Task 1 output", output_type="txt")


if __name__ == "__main__":
    run_task1("./data/task1/img5.jpg", None)
