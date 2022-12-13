#This code is useful to test Anime face detection in, for example, manga books you might have
#This was trained with images from google and realized data augmentation in order to find different features in different animes

# I might upload an explanation to the dataset, still wondering

import cv2

cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

manimeFaceClass = cv2.CascadeClassifier('PY to Test\cascade_BIG_IMG.xml')

while True:
	
	ret,frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	toy = manimeFaceClass.detectMultiScale(gray,
	scaleFactor = 1.3,
	minNeighbors = 99,
	minSize=(70,78))

	for (x,y,w,h) in toy:
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		cv2.putText(frame,'Anime Face',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)

	cv2.imshow('frame',frame)
	
	if cv2.waitKey(1) == 27:
		break
cap.release()
cv2.destroyAllWindows()
