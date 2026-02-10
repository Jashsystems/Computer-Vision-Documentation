import cv2 as cv
import numpy as np
img = cv.imread('../Images/park.jpg')
cv.imshow('img',img)

## Gray scale conversion ##
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

## Blur ##
for k in [3,7,15]:
    blur = cv.GaussianBlur(img, (k,k), 5)
    cv.imshow(f'Blur {k}', blur)
## Edge Cascade ##

for t1, t2 in [(50,100),(100,200)]:
    edges = cv.Canny(blur, t1, t2)
    cv.imshow(f"Canny {t1}-{t2}", edges)

## Dilating the image ##
for i in [1, 2, 3]:
    d = cv.dilate(edges, np.ones((3,3),np.uint8), iterations=i)
    cv.imshow(f"Dilate {i}", d)
##


