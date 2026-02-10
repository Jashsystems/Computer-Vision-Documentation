import cv2 as cv
import numpy as np
img = cv.imread('../Images/cat.jpg')
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

cv.waitKey(0)
if img is None:
    print("Image not found")
    exit()
