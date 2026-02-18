import cv2 as cv
import numpy as np

img = cv.imread('../Images/park.jpg')
cv.imshow('Park', img)

blank = np.zeros(img.shape[:2], dtype="uint8")

b,g,r = cv.split(img)

cv.imshow("Blue",b)
cv.imshow("Green",g)
cv.imshow("Red",r)

blue = cv.merge([b,blank,blank])
green= cv.merge([blank,g,blank])
red = cv.merge([blank,blank,r])

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

cv.imshow("b", blue)
cv.imshow("g", green)
cv.imshow("r", red)

merged_image = cv.merge([b,g,r])
cv.imshow("merged_image", merged_image)

cv.waitKey(0)