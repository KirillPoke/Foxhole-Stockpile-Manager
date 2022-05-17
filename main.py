import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'F:\\Coding\\tesseract.exe'
import matplotlib.pyplot as plt
import numpy as np
from pytesseract import Output
import os
import pandas as pd
from tqdm import tqdm


def main(psm, digits, use_padding):
    df_labels = pd.read_csv('labels_gold.csv').drop_duplicates()
    labels_dict = dict(zip(df_labels['filename'], df_labels['label']))

    dir = 'unrecognized'
    margin = 4
    if not digits:
        custom_config = f'--oem 3 --psm {psm}'
    else:
        custom_config = f'--oem 3 --psm {psm} outputbase digits'

    separator_digit = '39900993'
    separator_img = cv2.imread('separator2.png')
    separator_img = cv2.cvtColor(separator_img, cv2.COLOR_BGR2GRAY)
    separator_img = cv2.bitwise_not(separator_img)
    left_marg, right_marg, top_marg, bottom_marg = find_margins(separator_img)
    separator_img = separator_img[top_marg - margin:bottom_marg + margin, left_marg - margin:right_marg + margin]

    acc_list = []
    for file in [f for f in os.listdir(dir) if f != '.DS_Store']:
        true_label = labels_dict[file]
        img = cv2.imread(f'{dir}/{file}')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)

        left_marg, right_marg, top_marg, bottom_marg = find_margins(img)
        img = img[top_marg - margin:bottom_marg + margin, left_marg - margin:right_marg + margin]

        separator_img_resized = image_resize(separator_img, height=img.shape[0])

        if use_padding:
            # concatenate with separator image
            img = np.concatenate([separator_img_resized, img, separator_img_resized], axis=1)

        res = pytesseract.image_to_string(img, config=custom_config)

        res = res.strip().replace('\n', '')

        answer = res.replace(separator_digit, '').strip()

        acc_list.append(int(answer == true_label))


    return np.mean(acc_list)


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


if __name__ == '__main__':
    res_dict = {'psm': [], 'accuracy': []}
    psm = 7
    acc = main(psm, False, True)
    res_dict['psm'].append(psm)
    res_dict['accuracy'].append(acc)
    print(acc)
    pd.DataFrame(res_dict).to_csv('grid_search_res.csv', index=False)
