import cv2 as cv

def rescale_frame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)

img = cv.imread('../Images/cat.jpg')
image_resized = rescale_frame(img,0.8)
cv.imshow('Image_original',img)
cv.imshow('Image_rescaled',image_resized)
cv.waitKey(0)