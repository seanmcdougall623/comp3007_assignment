import os
from pathlib import Path

from datasets import augment_img

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

root_dir = Path(os.path.dirname(__file__))


# implementation inspired by https://github.com/shanlau/SVM_Recognizing_Digit/blob/master/svm_model.py


def load_image(img, size, loaded=False):
    if not loaded:
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # cvt grayscale if not alr
    elif len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize to reduce feature length
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    # threshold just to remove the segment noise or any light pollution
    digit = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    # dilate slightly
    digit = cv2.dilate(digit, np.ones((2, 2), np.uint8), iterations=1)

    # cv2.imshow("Digit", digit)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return digit


def load_images(img_path, size=(64, 64)):
    train_data = []
    labels = []
    for foldername in sorted(os.listdir(img_path)):
        for filename in sorted(os.listdir(img_path / foldername)):
            if filename.endswith(".png"):
                digit = load_image(
                    img_path / foldername / filename, size=size
                )

                train_data.append(digit)
                labels.append(int(foldername))

    # convert to np array and return
    return (
        np.asarray(train_data, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
    )


def train_svm(
    train_data,
    train_labels,
    size,
    model_dir,
):
    # create labels and reshape to 1d array
    train_data = np.reshape(train_data, (train_data.shape[0], size))

    # create and train the model!

    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC)
    model.setKernel(cv2.ml.SVM_LINEAR)
    model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.0e-06))
    model.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    model.save(model_dir)


# calculate training accuracy
def validate_svm(
    model_dir,
    test_data,
    test_labels,
    size,
    svm_matrix_name="svm_confusion_matrix.png",
):
    test_data = np.reshape(test_data, (test_data.shape[0], size))
    model = cv2.ml.SVM_load(model_dir)
    _, pred = model.predict(test_data)

    # reshape pred to 1d array
    pred = pred.reshape(-1).astype(int)

    # calc accuracy
    print(f"Accuracy: {np.mean(pred == test_labels) * 100}%")

    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("SVM Confusion Matrix")

    plt.savefig(svm_matrix_name)


def validate_with_gen_data(
    img_dir,
    model_dir,
    size=(64, 64),
    svm_matrix_name="svm_confusion_matrix.png",
):
    # augment training data with 5 imgs
    img_dir = root_dir / "lcd_digits"
    model_dir = root_dir / "../models/bp_svm.xml"

    validate_imgs = []
    validate_labels = []

    for img in sorted(img_dir.iterdir()):
        if img.is_file() and img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            augmented_imgs = augment_img(
                img,
                multiplier=5,
                scale=(0.8, 1.0),
                skew=(-0.1, 0.1),
            )
            for a_img in augmented_imgs:
                digit = load_image(a_img, size=size, loaded=True)
                validate_imgs.append(digit)
                validate_labels.append(int(img.stem))

    validate_svm(
        model_dir,
        np.asarray(validate_imgs, dtype=np.float32),
        np.asarray(validate_labels, dtype=np.int32),
        size[0] * size[1],
        svm_matrix_name=svm_matrix_name,
    )


def validate_bp():
    img_dir = root_dir / "lcd_digits"
    model_dir = root_dir / "../models/bp_svm.xml"
    validate_with_gen_data(
        img_dir,
        model_dir,
        size=(64, 64),
        svm_matrix_name="bp_svm_confusion_matrix.png",
    )


def validate_therm():
    img_dir = root_dir / "therm_digits"
    model_dir = root_dir / "../models/therm_svm.xml"
    validate_with_gen_data(
        img_dir,
        model_dir,
        size=(64, 64),
        svm_matrix_name="therm_svm_confusion_matrix.png",
    )


def train_bp():
    img_dir = root_dir / "../datasets/task3/lcd"
    model_dir = root_dir / "../models/bp_svm.xml"
    loaded_images, loaded_labels = load_images(img_dir)
    train_svm(
        loaded_images,
        loaded_labels,
        64 * 64,
        model_dir,
    )


def train_therm():
    img_dir = root_dir / "../datasets/task3/therm"
    model_dir = root_dir / "../models/therm_svm.xml"
    loaded_images, loaded_labels = load_images(img_dir, size=(64, 64))
    train_svm(
        loaded_images,
        loaded_labels,
        64 * 64,
        model_dir,
    )


def test_digit(img_path, model_dir, size=(64, 64), loaded=False):
    d_size = size[0] * size[1]
    digit = load_image(
        img_path,
        size,
        loaded=loaded,
    )

    # cv2.imshow("Digit", digit)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    model = cv2.ml.SVM_load(model_dir)

    sample = np.array([digit], dtype=np.float32).reshape((1, d_size))

    _, pred = model.predict(sample)

    return int(pred[0][0])


def test_bp_digit(img_path, loaded=False):
    model_dir = root_dir / "../models/bp_svm.xml"
    return test_digit(
        img_path,
        model_dir,
        loaded=loaded,
    )


def test_therm_digit(img_path, loaded=False):
    model_dir = root_dir / "../models/therm_svm.xml"
    return test_digit(img_path, model_dir, size=(64, 64), loaded=loaded)


if __name__ == "__main__":
    # train_bp()
    # train_therm()
    validate_bp()
    validate_therm()
    # bp = test_bp_digit(os.getcwd() + "/output/task2/d2.png")
    # therm = test_therm_digit(os.getcwd() + "/lib/therm_digits/4.png")
    # print(bp)
    # print(therm)
