import os

from roboflow import Roboflow
from utils import load_dotenv


def load_model():
    load_dotenv()

    rf = Roboflow(api_key=os.environ["ROBOFLOW_SECRET"])
    project = rf.workspace("sean-mcdougall").project(
        "bp-therm-identification"
    )
    project.version(1).download(
        "yolov8", location="./datasets/task1", overwrite=True
    )


if __name__ == "__main__":
    load_model()
