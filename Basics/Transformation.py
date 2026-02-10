import cv2 as cv
import numpy as np

img = cv.imread('../Images/park.jpg')
cv.imshow('Park', img)

# Translation
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# -x --> Left
# -y --> Up
# x --> Right
# y --> Down

translated = translate(img, -100, 100)
# translate(img, 100, 0)
# Move up
# translate(img, 0, -100)
# Diagonal
# translate(img, 50, -50)

cv.imshow('Translated', translated)
cv.waitKey(0)