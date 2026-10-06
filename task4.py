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

from task1 import extract_item, identify_object
from task2 import (
    extract_lcd_digits,
    extract_thermo_section,
    find_lcd_digits,
    find_thermo_section,
)
from task3 import get_digit_num, get_therm_reading


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


def run_task4(image_path, config):
    input_path = Path(image_path)
    for file in sorted(input_path.iterdir()):
        if file.is_file() and file.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
        ]:
            out = identify_object(file)
            if not out:
                print(f"Skipping {file.name}, no identifiable item found.")
                continue
            item = out[0]

            print(f"Identified {item[0]} in {file.name} with confidence {item[1]:.2f}")

            extracted_img = extract_item(file, item[2], item[0])

            txt_out = ""

            if item[0] == "bp":
                ref_cnt, img = find_lcd_digits(extracted_img, loaded=True, draw=False)
                d_out = extract_lcd_digits(img, ref_cnt)

                readings = []

                for digit in d_out:
                    num = get_digit_num(digit, loaded=True)
                    readings.append(str(num))
                if len(readings) != 7:
                    print(
                        f"Warning: Expected 7 readings, got {len(readings)} for {file.name}."
                    )

                output_path = f"output/task4/{file.stem}.txt"
                txt_out = f"bpm {''.join(readings[0:3])}, {''.join(readings[3:5])}, {''.join(readings[5:7])}"
                save_output(output_path, txt_out, output_type="txt")

            elif item[0] == "therm":
                # next to no clue why
                # but thermometer extraction only works when it reads it in itself
                # reading from the buffer doesn't seem to work as well
                temp_out = f"output/task4/{file.stem}_temp.jpg"
                cv2.imwrite(temp_out, extracted_img)
                box, img = find_thermo_section(temp_out)
                t_out = extract_thermo_section(img, box)
                os.remove(temp_out)
                reading = get_therm_reading(t_out, loaded=True)

                output_path = f"output/task4/{file.stem}.txt"
                txt_out = f"temp {reading}"

                save_output(output_path, txt_out, output_type="txt")

            print(f"Output for {file.name}: {txt_out}")

        else:
            print(f"Skipping {file.name}, not a valid image file.")


# debug purposes
if __name__ == "__main__":
    run_task4("data/task1/extended", None)
