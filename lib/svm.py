import os
from pathlib import Path

import cv2
import numpy as np
from utils import resize

root_dir = os.path.dirname(__file__)
img_dir = root_dir + "/lcd_digits/"
model_dir = str(Path(root_dir).parent) + "/models/digit_svm.xml"


# TODO: find a 0 LCD segment cause you could be dead


# implementation inspired by https://github.com/shanlau/SVM_Recognizing_Digit/blob/master/svm_model.py


def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # resize to 64x64 to reduce feature length
    img = resize(img, (64, 64))

    # threshold just to remove the segment noise or any light pollution
    digit = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return digit


def load_images():
    train_data = []
    for filename in sorted(os.listdir(img_dir)):
        if filename.endswith(".png"):

            digit = load_image(img_dir + filename)

            train_data.append(digit)

    # convert to np array and return
    return np.asarray(train_data, dtype=np.float32)


def train_svm(train_data):
    # 9 digits, each with 64*64=4096 features
    train_data = np.reshape(train_data, (9, 64 * 64))
    train_label = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)

    # create and train the model!

    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC)
    model.setKernel(cv2.ml.SVM_LINEAR)
    model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.0e-06))
    model.train(train_data, cv2.ml.ROW_SAMPLE, train_label)

    model.save(model_dir)


def test_digit(img_path):
    digit = load_image(img_path)
    model = cv2.ml.SVM_load(model_dir)

    sample = np.array([digit], dtype=np.float32).reshape((1, 4096))

    _, pred = model.predict(sample)

    return int(pred[0][0])


if __name__ == "__main__":
    digit = test_digit(os.getcwd() + "/data/task3/lcd4/d7.png")
    print(digit)
