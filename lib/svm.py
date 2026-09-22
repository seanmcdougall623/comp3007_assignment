import os
from pathlib import Path

import cv2
import numpy as np

# can be calledfrom both task3.py and here, so handle import errors
try:
    from utils import resize
except ImportError:
    from lib.utils import resize

root_dir = os.path.dirname(__file__)

# TODO: find a 0 LCD segment cause you could be dead
# or just anyway BP reading with a 0 in it


# implementation inspired by https://github.com/shanlau/SVM_Recognizing_Digit/blob/master/svm_model.py


def load_image(img, size, loaded=False):
    if not loaded:
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    # resize to reduce feature length
    img = resize(img, size)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    # threshold just to remove the segment noise or any light pollution
    digit = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[
        1
    ]
    return digit


def load_images(root_dir, size=(64, 64)):
    train_data = []
    for filename in sorted(os.listdir(root_dir)):
        if filename.endswith(".png"):
            digit = load_image(root_dir + filename, size=size)

            train_data.append(digit)

    # convert to np array and return
    return np.asarray(train_data, dtype=np.float32)


def train_svm(train_data, digit_count, size, model_dir, start_label=0):
    # create labels and reshape to 1d array
    train_data = np.reshape(train_data, (digit_count, size))
    train_label = np.arange(
        start_label, start_label + digit_count, dtype=np.int32
    )
    # create and train the model!

    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC)
    model.setKernel(cv2.ml.SVM_LINEAR)
    model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.0e-06))
    model.train(train_data, cv2.ml.ROW_SAMPLE, train_label)

    model.save(model_dir)


def train_bp():
    img_dir = root_dir + "/lcd_digits/"
    model_dir = str(Path(root_dir).parent) + "/models/bp_svm.xml"
    loaded_images = load_images(img_dir)
    train_svm(loaded_images, 9, 64 * 64, model_dir, start_label=1)


def train_therm():
    img_dir = root_dir + "/therm_digits/"
    model_dir = str(Path(root_dir).parent) + "/models/therm_svm.xml"
    loaded_images = load_images(img_dir, size=(32, 32))
    train_svm(loaded_images, 8, 32 * 32, model_dir, start_label=-2)


def test_digit(img_path, model_dir, size=(64, 64), loaded=False):
    d_size = size[0] * size[1]
    digit = load_image(
        img_path,
        size,
        loaded=loaded,
    )
    model = cv2.ml.SVM_load(model_dir)

    sample = np.array([digit], dtype=np.float32).reshape((1, d_size))

    _, pred = model.predict(sample)

    return int(pred[0][0])


def test_bp_digit(img_path, loaded=False):
    model_dir = str(Path(root_dir).parent) + "/models/bp_svm.xml"
    return test_digit(img_path, model_dir, loaded=loaded)


def test_therm_digit(img_path, loaded=False):
    model_dir = str(Path(root_dir).parent) + "/models/therm_svm.xml"
    return test_digit(img_path, model_dir, size=(32, 32), loaded=loaded)


if __name__ == "__main__":
    train_bp()
    train_therm()
    bp = test_bp_digit(os.getcwd() + "/data/task3/lcd4/d7.png")
    therm = test_therm_digit(os.getcwd() + "/lib/therm_digits/1.png")
    print(bp)
    print(therm)
