import cv2
from deepface import DeepFace

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1)

while True:
    ret, img = cap.read()
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame_gray, 1.3, 5)
    response = DeepFace.analyze(img, actions=("emotion",), enforce_detection=False)
    print(response)
    
    for face in faces:
        x, y, w, h = face
        cv2.putText(img, text = response["dominant_emotion"], org=(x,y), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (0, 0, 255))
        rectangleFrame = cv2.rectangle(img, (x, y), (x+w, y+h), color = (255, 0, 0), thickness =3)
    
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:      #Press Esc to stop the video
        break
    
cap.release()
cv2.destroyAllWindows()