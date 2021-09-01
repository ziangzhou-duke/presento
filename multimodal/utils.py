import cv2
import numpy as np


def blend_image(img1: np.ndarray, img2: np.ndarray):
    """
    Blend img2 onto img1

    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html#bitwise-operations
    """
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst

    return img1


def draw_annotation(image: np.ndarray, x: float, y: float, text: str):
    # draw label
    label_size, base_line = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    )
    cv2.rectangle(
        image,
        (x, y + 1 - label_size[1]),
        (x + label_size[0], y + 1 + base_line),
        (223, 128, 255),
        cv2.FILLED,
    )
    cv2.putText(
        image,
        text,
        (x, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
    )
