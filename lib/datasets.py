import os
import shutil
from pathlib import Path

import kagglehub
from roboflow import Roboflow
from utils import load_dotenv


def get_rb_dataset():
    rf = Roboflow(api_key=os.environ["ROBOFLOW_SECRET"])
    project = rf.workspace("sean-mcdougall").project("bp-therm-identification")
    project.version(4).download(
        "yolov8-obb", location="./datasets/task1", overwrite=True
    )


def get_digit_dataset():
    path = kagglehub.dataset_download(
        "cramatsu/7-segment-display-yolov8",
        output_dir="./datasets/task3",
    )
    # kagglehub does smthing weird with file structure (stupid)
    folder = Path(path + "/segment_indigator.v4i.yolov8")
    for item in folder.iterdir():
        destination = folder.parent / item.name
        shutil.move(str(item), str(destination))

    os.rmdir(folder)


def load_datasets():
    load_dotenv()
    get_digit_dataset()
    get_rb_dataset()


if __name__ == "__main__":
    get_digit_dataset()
