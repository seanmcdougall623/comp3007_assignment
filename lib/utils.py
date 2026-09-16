import cv2


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
