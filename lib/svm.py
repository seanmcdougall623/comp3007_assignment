import os
from pathlib import Path

import cv2
import numpy as np

# can be calledfrom both task3.py and here, so handle import errors
try:
    from utils import resize
except ImportError:
    from lib.utils import resize

root_dir = Path(os.path.dirname(__file__))

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


def load_images(img_path, size=(64, 64)):
    train_data = []
    label_count = 0
    data_count = 0
    labels = []
    print(img_path)
    for foldername in sorted(os.listdir(img_path)):
        for filename in sorted(os.listdir(img_path / foldername)):
            if filename.endswith(".png"):
                digit = load_image(
                    img_path / foldername / filename, size=size
                )

                train_data.append(digit)
                labels.append(int(foldername))
                data_count += 1
        label_count += 1

    # convert to np array and return
    return (
        np.asarray(train_data, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        label_count,
        data_count // label_count,
    )


def train_svm(
    train_data,
    train_labels,
    digit_count,
    data_count,
    size,
    model_dir,
):
    # create labels and reshape to 1d array
    train_data = np.reshape(train_data, (digit_count * data_count, size))

    # normalise the data
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    # avoid div by 0
    std[std == 0] = 1.0
    train_data = (train_data - mean) / std
    # create and train the model!

    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC)
    model.setKernel(cv2.ml.SVM_LINEAR)
    model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.0e-06))
    model.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    model.save(model_dir)

    # save the noramlisation consts!
    with open(model_dir.with_suffix(".txt"), "w") as f:
        f.write(f"{mean.tolist()}\n")
        f.write(f"{std.tolist()}\n")


def train_bp():
    img_dir = root_dir / "../datasets/task3/lcd"
    model_dir = root_dir / "../models/bp_svm.xml"
    loaded_images, loaded_labels, label_count, data_count = load_images(
        img_dir
    )
    train_svm(
        loaded_images,
        loaded_labels,
        label_count,
        data_count,
        64 * 64,
        model_dir,
    )


def train_therm():
    img_dir = root_dir / "../datasets/task3/therm"
    model_dir = root_dir / "../models/therm_svm.xml"
    loaded_images, loaded_labels, label_count, data_count = load_images(
        img_dir, size=(32, 32)
    )
    train_svm(
        loaded_images,
        loaded_labels,
        label_count,
        data_count,
        32 * 32,
        model_dir,
    )


def test_digit(
    img_path, model_dir, size=(64, 64), loaded=False, mean=None, std=None
):
    d_size = size[0] * size[1]
    digit = load_image(
        img_path,
        size,
        loaded=loaded,
    )
    model = cv2.ml.SVM_load(model_dir)

    sample = np.array([digit], dtype=np.float32).reshape((1, d_size))

    if mean is not None and std is not None:
        # avoid div by 0
        std[std == 0] = 1.0
        sample = (sample - mean) / std

    _, pred = model.predict(sample)

    return int(pred[0][0])


def test_bp_digit(img_path, loaded=False):
    model_dir = root_dir / "../models/bp_svm.xml"
    with open(model_dir.with_suffix(".txt"), "r") as f:
        mean = np.array(eval(f.readline().strip()), dtype=np.float32)
        std = np.array(eval(f.readline().strip()), dtype=np.float32)
    return test_digit(
        img_path,
        model_dir,
        loaded=loaded,
        mean=mean,
        std=std,
    )


def test_therm_digit(img_path, loaded=False):
    model_dir = root_dir / "../models/therm_svm.xml"
    with open(model_dir.with_suffix(".txt"), "r") as f:
        mean = np.array(eval(f.readline().strip()), dtype=np.float32)
        std = np.array(eval(f.readline().strip()), dtype=np.float32)
    return test_digit(
        img_path, model_dir, size=(32, 32), loaded=loaded, mean=mean, std=std
    )


if __name__ == "__main__":
    train_bp()
    train_therm()
    bp = test_bp_digit(os.getcwd() + "/output/task2/d2.png")
    therm = test_therm_digit(os.getcwd() + "/lib/therm_digits/1.png")
    print(bp)
    print(therm)
