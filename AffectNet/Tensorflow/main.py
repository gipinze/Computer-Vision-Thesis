from keras.models import load_model
from time import sleep
from keras.utils.image_utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

Path_text = f'T:/Thesis project/Python Projects/Thesis Files/FER2013/Face Detection Models/deploy.prototxt.txt'
Path_caffee = f'T:/Thesis project/Python Projects/Thesis Files/FER2013/Face Detection Models/res10_300x300_ssd_iter_140000_fp16.caffemodel'

face_classifier=cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')
# face_classifier = cv2.dnn.readNetFromCaffe(Path_text, Path_caffee)
emotion_model = load_model('emotion_detection_model_50epochs.h5')


class_labels=['Angry','Contempt','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    labels=[]
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        #Get image ready for prediction
        roi=roi_gray.astype('float')/255.0  #Scale
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)  #Expand dims to get it ready for prediction (1, 48, 48, 1)

        preds=emotion_model.predict(roi)[0]  #Yields one hot encoded result for 8 classes
        label=class_labels[preds.argmax()]  #Find the label
        label_position=(x,y)
        cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
