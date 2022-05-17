import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'F:\Coding\\tesseract.exe'
#img = cv2.imread('C:\\Users\\Kirill\\Desktop\\stockpile.png')  # read image


def get_number_contours(img):
    # Gel all pixels with the color of the stockpile number
    gray = np.all(img == (98, 98, 98), 2)  # gray is a logical matrix with True where BGR = (34, 33, 33).
    # Find contours
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
        -2]  # Use index [-2] to be compatible to OpenCV 3 and 4
    filtered_cnts = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 20 and w > 30:  # Ignore gray pixels
            # Cropping the text block for giving input to OCR
            cropped = img[y:y + h, x:x + w]
            filtered_cnts.append(cnts)
            # Apply OCR on the cropped image
            # text = pytesseract.image_to_string(cropped, config='--psm 8 outputbase digits')
            # letters = string.ascii_lowercase
            # file_name = ''.join(random.choice(letters) for i in range(10))
            # if text:
            #     texts.append(text)
            #     recognized.append(cropped)
            #     cv2.imwrite(f'C:\\Users\Kirill\\Desktop\\recognized\\{file_name}.png', cropped)
            #     # cv2.imshow('crop', cropped)
            #     # cv2.waitKey()
            #     # cv2.destroyAllWindows()
            # else:
            #     cv2.imwrite(f'C:\\Users\Kirill\\Desktop\\recognized\\{file_name}.png', cropped)
