import ultralytics

ultralytics.checks()
import os
import shutil

from ultralytics import YOLO


def train_model(dataset_path, output_dir, epochs=100, m_type="obb"):
    if m_type == "obb":
        model = YOLO("yolov8n-obb.pt")
    elif m_type == "normal":
        model = YOLO("yolov8n.pt")
    elif m_type == "cls":
        model = YOLO("yolov8n-cls.pt")
    else:
        raise AssertionError("Please specify either 'obb' or 'normal' or 'cls'")

    results = model.train(
        data=os.getcwd() + dataset_path,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0,
    )

    # copy model out of runs folder to we can sync it to git
    # if statement more just to supress intellisense error
    if results:
        shutil.move(
            str(os.getcwd()) + "/" + str(results.save_dir) + "/weights/best.pt",
            str(os.getcwd()) + "/models" + "/" + output_dir,
        )


def train_bp_therm():
    train_model("/dataset/task1/data.yaml", "bp_therm_ident.pt")


def train_digit():
    train_model("/dataset/task3/data.yaml", "digit_cls.pt", m_type="cls")


if __name__ == "__main__":
    train_model()
