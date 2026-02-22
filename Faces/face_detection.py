import cv2 as cv


img = cv.imread('../Images/group_4.jpg')
cv.imshow('Group of people',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray scaled image of group',gray)

haar_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)

print(f"Number of faces detected: {len(faces_rect)}")
print(faces_rect)

for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv.imshow('Face detection result',img)

cv.waitKey(0)