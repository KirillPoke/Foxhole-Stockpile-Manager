import cv2
import pytesseract

from image_finder import get_number_contours, get_storage_header

pytesseract.pytesseract.tesseract_cmd = r'F:\\Coding\\tesseract.exe'
import numpy as np


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def find_margins(img):
    _, img = cv2.threshold(np.array(img), 100, 255, cv2.THRESH_BINARY)
    indicators_h = [idx for idx, val in enumerate(img.mean(axis=0) == 255) if val == False]
    indicators_v = [idx for idx, val in enumerate(img.mean(axis=1) == 255) if val == False]
    leftmost = min(indicators_h)
    rightmost = max(indicators_h)
    top = min(indicators_v)
    bottom = max(indicators_v)
    return leftmost, rightmost, top, bottom


margin = 4


def add_padding(img):
    separator_img = cv2.imread('separator2.png')
    separator_img = cv2.cvtColor(separator_img, cv2.COLOR_BGR2GRAY)
    separator_img = cv2.bitwise_not(separator_img)
    left_marg, right_marg, top_marg, bottom_marg = find_margins(separator_img)

    separator_img = separator_img[top_marg - margin:bottom_marg + margin, left_marg - margin:right_marg + margin]
    separator_img_resized = image_resize(separator_img, height=img.shape[0])
    padded_img = np.concatenate([separator_img_resized, img, separator_img_resized], axis=1)
    return padded_img


def pre_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)

    left_marg, right_marg, top_marg, bottom_marg = find_margins(img)
    img = img[top_marg - margin:bottom_marg + margin, left_marg - margin:right_marg + margin]
    return img


def validate_padding_found(string, padding_string):
    if string.count(padding_string) != 2:
        print("Did not found correct padding")
        #raise AssertionError("Did not found correct padding")


def fancy_img_to_string(img):
    separator_digit = '39900993'
    custom_config = f'--oem 3 --psm 7'
    pre_processed_img = pre_process(img)
    padded_img = add_padding(pre_processed_img)
    string_from_image = pytesseract.image_to_string(padded_img, config=custom_config)
    validate_padding_found(string_from_image, separator_digit)
    string_from_image = string_from_image.strip().replace('\n', '')
    string_from_image = string_from_image.replace(separator_digit, '').strip()
    return string_from_image


if __name__ == '__main__':
    dir = 'unrecognized'
    img = cv2.imread('C:\\Users\\Kirill\\Desktop\\stockpile.png')
    contours = get_number_contours(img)
    for cnt in contours:
        string_from_image = fancy_img_to_string(cnt)
        print(string_from_image)
