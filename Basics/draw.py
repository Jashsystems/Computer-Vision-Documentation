import cv2 as cv
import numpy as np

Blank = np.zeros((500,500,3), dtype='uint8')
#cv.imshow('Blank',Blank)
# Paint the image a certain color ##
#Blank[100:200,300:500] = 255,0,0
#cv.imshow('Blank',Blank)
#cv.waitKey(0)

## Draw a Rectangle ##
#cv.rectangle(Blank,(0,0),(Blank.shape[1]//2,Blank.shape[0]//2),(0,0,255),1)
#cv.imshow('Rectangle',Blank)
#cv.waitKey(0)

## Draw a cirle ##
#cv.circle(Blank,(Blank.shape[1]//2, Blank.shape[0]//2),100, (0,0,255), thickness=2)
#cv.imshow("Circle",Blank)
#cv.waitKey(0)

## Draw a Line ##
#cv.line(Blank,(Blank.shape[1]//8,0),(300,500),(255,0,0),1)
#cv.imshow("Line",Blank)
#cv.waitKey(0)

## Write text ##
cv.putText(Blank,"Let it happen by Tame Impala",(100,200),cv.FONT_HERSHEY_TRIPLEX,0.5,(0,255,0),1)
cv.imshow("text",Blank)
cv.waitKey(0)
## Way to fit the entire text in the window ##