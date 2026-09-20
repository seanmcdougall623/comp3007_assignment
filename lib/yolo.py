import ultralytics

ultralytics.checks()
from ultralytics import YOLO


def train_model():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="task1/data.yaml",
        epochs=300,
        imgsz=640,
        batch=16,
        device=0,
    )


if __name__ == "__main__":
    train_model()
