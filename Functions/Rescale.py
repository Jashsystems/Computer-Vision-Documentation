import cv2 as cv

def rescale_frame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)

capture = cv.VideoCapture('../Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()

    if not isTrue:
        break   # 👈 absolutely mandatory

    video_frame = rescale_frame(frame, 0.75)

    cv.imshow('Video_frame', video_frame)
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
