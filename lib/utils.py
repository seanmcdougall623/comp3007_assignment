import math
import os

import cv2
import numpy as np


# helper function for ordering points
# used for task 1
# inspired from https://stackoverflow.com/a/79378245
def order_corners(corners):
    pts = corners.copy()
    cx, cy = np.mean(pts, axis=0)

    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)

    sorted_idx = np.argsort(angles)
    pts_sorted = pts[sorted_idx]

    return pts_sorted


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


# helper method to extract digits
# used for both task 2 adn 3
def extract_digits(
    img,
    kernel_size=(5, 5),
    threshold1=5,
    threshold2=120,
    draw=False,
    area_threshold=2000,
):
    # thresholding
    img_edges = cv2.Canny(img, threshold1, threshold2)

    # dilate the threshold a little to connect the segments
    kernel = np.ones(kernel_size, np.uint8)
    img_dil = cv2.dilate(img_edges, kernel, iterations=1)

    contours, _ = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ref_cnt = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # check if sufficiently big, and rectangular shape
        if area > area_threshold and (w * 1.3 < h):
            # if we want to draw use direct contour approximation
            # used for drawing the output if required
            if draw:
                ref_cnt.append(
                    np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                )
            else:
                # much more computational efficent
                ref_cnt.append((x, y, w, h))

    # sort based on pixel position
    # take into account both x and y
    # idk probably not efficient but shld be fine
    # only sort if we aren't drawing cause otherwise this method breaks and drawing is really just an diagnostic tool anyway
    if not draw:
        ref_cnt = sorted(ref_cnt, key=lambda r: math.sqrt(r[0] ** 2 + r[1] ** 2))

    return ref_cnt


# extracts coordinates of largest red region in image
# used to computer thermometer info
# @param loaded_img must be already loaded opencv2 img
def extract_red(loaded_img):
    img_hsv = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2HSV)

    # !!!
    # TODO: REFERENCE: AI GENERATED
    # !!!
    lower = np.array([5, 55, 90])
    upper = np.array([25, 255, 200])
    # ai stuff ends
    blur = cv2.GaussianBlur(img_hsv, (3, 3), 0)
    red_mask = cv2.inRange(blur, lower, upper)
    # use morphological filter to filter out noise
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # work out where largest blob is and grab it
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # grab coords of where the box is
    x, y, w, h, _ = stats[largest]

    return (x, y, w, h)


# resizes an image a bit smarter by padding it out if one dimension is smaller than requested
# used for digit recognition as default cv2 resizing tends to blow up the '1' and make it hard to recognise
# look the SVM can probably learn it but better safe than sorry
# @param img - loaded cv2 img
# @param dims - (w,h) of target sizing
def resize(img, dims):
    tar_w, tar_h = dims

    h, w = img.shape
    if h < tar_h or w < tar_w:
        pad_h = max(0, tar_h - h)
        pad_w = max(0, tar_w - w)

        # center padding
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255
        )

    return cv2.resize(img, (tar_w, tar_h), cv2.INTER_AREA)


# calculates the digit from some lcd segment mathemtically
# much more robust, not allowed by assignment guidelines :(
# using as backup plan if classifier doesn't work
# inspired by https://pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
# rather than doing some fancy model we can instead iterate a segment over our segment display
# if the segment is 'on' then record it, otherwise don't
# use a predefined lookup table to work out which on segments correspond to which digit

DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}


def get_digit_num_manual(digit):

    # number 1 digit will only have 2 segments - computation will fail
    # just count number of countours and if we only have two its probs a 1
    num_cnts, _ = cv2.findContours(digit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(num_cnts) == 2:
        return 1

    # precomputed width and height of segments
    h, w = digit.shape
    dW, dH = (int(w * 0.25), int(h * 0.15))
    dHC = int(h * 0.05)

    segments = [
        ((0, 0), (w, dH)),  # top
        ((0, 0), (dW, h // 2)),  # top-left
        ((w - dW, 0), (w, h // 2)),  # top-right
        ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
        ((0, h // 2), (dW, h)),  # bottom-left
        ((w - dW, h // 2), (w, h)),  # bottom-right
        ((0, h - dH), (w, h)),  # bottom
    ]
    on = [0] * len(segments)

    for i, ((xA, yA), (xB, yB)) in enumerate(segments):
        segROI = digit[yA:yB, xA:xB]
        total = cv2.countNonZero(segROI)
        area = (xB - xA) * (yB - yA)
        if total / float(area) > 0.5:
            on[i] = 1

    digit = DIGITS_LOOKUP[tuple(on)]
    return digit
