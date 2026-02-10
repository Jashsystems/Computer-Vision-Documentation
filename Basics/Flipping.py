import cv2 as cv

img = cv.imread('../Images/park.jpg')
# Flipping
flip = cv.flip(img, -1)
cv.imshow('Flip', flip)




cv.imshow('Horizontal', cv.flip(img, 1))
cv.imshow('Vertical', cv.flip(img, 0))
cv.imshow('Both', cv.flip(img, -1))

cv.waitKey(0)