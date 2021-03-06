import cv2
import numpy as np

from utils import show_img


def get_number_contours(img):
    # Gel all pixels with the color of the stockpile number
    gray = np.all(img == (98, 98, 98), 2)
    gray = gray.astype(np.uint8) * 255
    get_storage_header(img)
    # gray is a logical matrix with True where BGR = (34, 33, 33).
    # Find contours
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
        -2]  # Use index [-2] to be compatible to OpenCV 3 and 4
    filtered_cnts = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 20 and w > 30:  # Ignore gray pixels
            # Cropping the text block for giving input to OCR
            cropped = img[y:y + h, x:x + w]
            filtered_cnts.append(cropped)
    return filtered_cnts

def get_storage_header(img):
    img = cv2.blur(img, (5, 5))
    header_color = (143, 144, 143)
    header = np.all(img == header_color, 2)
    header = header.astype(np.uint8) * 255
    show_img(header)