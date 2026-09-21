import os

import cv2
import numpy as np


# convert and filter out noise
def preprocess_image(img, resize=True, resize_dimensions=(600, 480), gauss=True):
    if resize:
        resized = cv2.resize(img, resize_dimensions, interpolation=cv2.INTER_LINEAR)
    else:
        resized = img
    grayed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    if gauss:
        blur = cv2.GaussianBlur(grayed, (5, 5), 0)
    else:
        blur = grayed

    return blur


# library definition doesn't have support for python-dotenv :(
# tiny little loader to grab our roboflow API key
# technically not needed at assessment time but whatever
def load_dotenv():
    with open(".env", "r") as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            os.environ[key] = value


# extracts coordinates of largest red region in image
# used to computer thermometer info
# @param loaded_img must be already loaded opencv2 img
def extract_red(loaded_img):
    img_hsv = cv2.cvtColor(loaded_img)

    # !!!
    # TODO: REFERENCE: AI GENERATED
    # !!!
    lower = np.array([5, 55, 90])
    upper = np.array([25, 255, 200])
    # ai stuff ends

    red_mask = cv2.inRange(img_hsv, lower, upper)
    # use morphological filter to filter out noise
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # work out where largest blob is and grab it
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # grab coords of where the box is
    x, y, w, h, _ = stats[largest]

    return (x, y, w, h)
