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
def preprocess_image(
    img, resize=True, resize_dimensions=(600, 480), gauss=True
):
    if resize:
        resized = cv2.resize(
            img, resize_dimensions, interpolation=cv2.INTER_LINEAR
        )
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
    # gaussian
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # thresholding
    img_edges = cv2.Canny(img, threshold1, threshold2)

    # dilate the threshold a little to connect the segments
    kernel = np.ones(kernel_size, np.uint8)
    img_dil = cv2.dilate(img_edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    img_w, img_h = img.shape[1], img.shape[0]

    ref_cnt = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # check if sufficiently big, and rectangular shape
        if (
            area > area_threshold
            and (w * 1.3 < h)
            # ignore anything super big as probs false positive
            and area < 0.5 * img_w * img_h
            # ignore anything too close to left and right
            and x > 0.05 * img_w
            and x + w < img_w - 0.05 * img_w
        ):
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
        ref_cnt = sorted(
            ref_cnt, key=lambda r: math.sqrt(r[0] ** 2 + r[1] ** 2)
        )

    return ref_cnt


# specifies extracting digits to just what we are interested in on the thermometer
# used for both task 2 and task 3
# @param requires img to be loaded already
def extract_therm_digits(img, area_threshold_scale=500, draw=False):
    b_w = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # crop down to left hand third of image so we only get the digits in celsius
    third_img = b_w[:, : b_w.shape[1] // 3]

    h, w = third_img.shape
    ref_cnt = extract_digits(
        third_img,
        kernel_size=(1, 1),
        area_threshold=h * w // area_threshold_scale,
        draw=draw,
    )

    if draw:
        cv2.drawContours(third_img, ref_cnt, -1, (0, 255, 0), 2)

        cv2.imshow("Thermometer Digits", third_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return ref_cnt, third_img


# used to check if any digits are negative
def check_negative(img, draw=False):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(img, 50, 150)

    _, img_w = img.shape[:2]

    cnt = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
        0
    ]

    ref_cnt = []

    for c in cnt:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > 100 and w > 2 * h:
            if draw:
                ref_cnt.append(
                    np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                )
            else:
                ref_cnt.append((x, y, w, h))

    if draw:
        cv2.drawContours(img, ref_cnt, -1, (0, 255, 0), 2)

        cv2.imshow("Negative Sign", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # return if contours near the left side
        if ref_cnt:
            for cnt in ref_cnt:
                x, y, w, h = cnt
                if x < 0.25 * img_w:
                    return True


# extracts coordinates of largest red region in image
# used to computer thermometer info
# @param loaded_img must be already loaded opencv2 img
def extract_red(loaded_img, a_filt=130):
    img_lab = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2LAB)

    # AI threshold found via Claude
    a = img_lab[:, :, 1]
    red_mask = np.where(a > a_filt, 255, 0).astype(np.uint8)

    # close gaps due to noise
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 50))
    mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, close_kernel)

    # use morphological filter to filter out noise
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

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
def resize(img, dims, width=True, height=False):
    tar_w, tar_h = dims

    h, w = img.shape
    if h < tar_h or w < tar_w:
        pad_h = max(0, tar_h - h)
        pad_w = max(0, tar_w - w)

        # center padding
        top = pad_h // 2 if height else 0
        bottom = pad_h - top if height else 0
        left = pad_w // 2 if width else 0
        right = pad_w - left if width else 0

        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255
        )

    return cv2.resize(img, (tar_w, tar_h), cv2.INTER_AREA)


def correct_skew(img, bb):

    # order in l->r, t->b
    # in case hasn't been done yet
    corners = order_corners(bb)

    tl, tr, br, bl = corners
    # implementation inspired by https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    # use linalg normalisation cuz thats less words
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    max_width = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    max_height = max(int(heightA), int(heightB))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    return warped, M


# Generated by Claude
def pad_bb(bb, px=50, py=50):
    tl, tr, br, bl = bb
    width_dir = tr - tl
    width_dir = width_dir / np.linalg.norm(width_dir)
    height_dir = bl - tl
    height_dir = height_dir / np.linalg.norm(height_dir)
    return np.array(
        [
            tl - width_dir * px - height_dir * py,
            tr + width_dir * px - height_dir * py,
            br + width_dir * px + height_dir * py,
            bl - width_dir * px + height_dir * py,
        ],
        dtype=np.float32,
    )


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
    num_cnts, _ = cv2.findContours(
        digit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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
