import cv2


def show_img(img):
    cv2.imshow('graycsale image', img)
    cv2.waitKey(0)