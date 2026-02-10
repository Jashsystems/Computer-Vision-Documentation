import cv2 as cv

#img = cv.imread('../Images/cat_large.jpg')

#cv.imshow('cat_large', img)
#cv.waitKey(0)

capture = cv.VideoCapture('../Videos/dog.mp4')
while True:
    isTrue, frame = capture.read()
    cv.imshow('frame', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()