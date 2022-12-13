########################################################

#Apply the above logic to a live video
import numpy as np
import cv2

# face_cascade = cv2.CascadeClassifier('cascade_DATA_AUG.xml')
face_cascade = cv2.CascadeClassifier('cascade4.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascades_models/haarcascade_eye.xml')

#Check if your system can detect camera and what is the source number
# cams_test = 10
# for i in range(0, cams_test):
#     cap = cv2.VideoCapture(i)
#     test, frame = cap.read()
#     print("i : "+str(i)+" /// result: "+str(test))


#There is a proble when we try to identify faces with the "small images", probably due to the small images used to train the cascade

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor = 1.3, 
        minNeighbors = 5)
    
    #First detect face and then look for eyes inside the face.
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
            
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:      #Press Esc to stop the video
        break

cap.release()
cv2.destroyAllWindows()