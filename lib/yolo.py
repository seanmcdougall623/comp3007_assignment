import ultralytics

ultralytics.checks()
from ultralytics import YOLO

import os
import shutil


def train_model(epochs=100):
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=os.getcwd() + "/datasets/task1/data.yaml",
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0,
    )

    # copy model out of runs folder to we can sync it to git
    # if statement more just to supress intellisense error
    if results:
        shutil.move(
            str(os.getcwd())
            + "/"
            + str(results.save_dir)
            + "/weights/best.pt",
            str(os.getcwd()) + "/models/bp_therm_ident.pt",
        )


if __name__ == "__main__":
    train_model()
