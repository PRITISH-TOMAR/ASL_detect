import numpy as np
import cv2
minValue = 70
def func(path):    
    frame = cv2.imread(path)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)

    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return res


import cv2
import numpy as np

minValue = 70

def func2(image):
    # Check if the image has 3 channels (color), otherwise do not apply cvtColor
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert the image to grayscale if it has 3 channels
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # If the image is already grayscale (1 channel), no need to convert
        gray = image
    
    # Apply Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    # Apply Adaptive Thresholding
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply standard Thresholding with OTSU method
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return res
