import cv2 as cv

img = cv.imread("../images/cats.jpg")
cv.imshow("Cats",img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray_image",gray)

# Thresholding
threshold, thresh = cv.threshold(gray,150,255,cv.THRESH_BINARY)
cv.imshow("thresh_image",thresh)
print(type(thresh))

threshold0, thresh_inv = cv.threshold(gray,150,255,cv.THRESH_BINARY_INV)
cv.imshow('thresh_inv_image',thresh_inv)

adaptive_thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,51,9)
cv.imshow("adaptive_thresh_image",adaptive_thresh)
cv.waitKey(0)