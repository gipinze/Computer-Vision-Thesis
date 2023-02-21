import cv2
import mediapipe as mp
import math

# Video capture

# The window, where the detection is going to happen, needs to be big enough to detect the mesh and the dots

cap = cv2.VideoCapture(0) # number will depends on the camera (0, 1, etc)
cap.set(3, 1280) # Windows width
cap.set(4, 720) # windows height

# Draw Function

mpDraw = mp.solutions.drawing_utils
confDraw = mpDraw.DrawingSpec(thickness = 1, circle_radius = 1) # adjusting the drawing conf

# create object to storage the facial mesh
mpFaceMesh = mp.solutions.face_mesh # Calling the function
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1) # create the object

while True: 

    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(frameRGB)

    # List to store the results
    px = []
    py = []
    list = []
    r = 5
    t = 3

    if results.multi_face_landmarks:
        for faces in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faces, mpFaceMesh. FACEMESH_CONTOURS, confDraw)

            for id, points in enumerate(faces.landmark):
                al, an, c = frame.shape
                x, y = int(points.x*an), int(points.y*al)
                px.append(x)
                py.append(y)
                list.append([id, x , y])
                if len(list) == 468:
                    #Right eyebrow
                    x1, y1 = list[65][1:]
                    x2, y2 = list[158][1:]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    # cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), t)
                    # cv2.circle(frame, (x1, y1), r, (0, 0, 0), cv2.FILLED) # This lines help us to understand what points are we going to modify and their distance
                    # cv2.circle(frame, (x2, y2), r, (0, 0, 0), cv2.FILLED)
                    # cv2.circle(frame, (x1, y1), r, (0, 0, 0), cv2.FILLED)
                    lenght1 = math.hypot(x2 - x1, y2 - y1)

                    # Left eyebrow
                    x3, y3 = list[295][1:]
                    x4, y4 = list[385][1:]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                    lenght2 = math.hypot(x4 - x3, y4 - y3)

                    # Mouth borders
                    x5 ,y5 = list[78][1:]
                    x6, y6 = list[308][1:]
                    cx3, cy3 = (x5 + x6) // 2, (y5 + y6) //2 
                    lenght3 = math.hypot(x6 - x5, y6 - y5)

                    # Mouth open
                    x7, y7 = list[13][1:]
                    x8, y8 = list[14][1:]
                    cx4, cy4 = (x7 + x8) // 2, (y7 + y8) // 2
                    lenght4 = math.hypot(x8 - x7, y8 - y7)

                    #Classification
                    #Angry
                    if lenght1 < 19 and lenght2 < 19 and lenght3 > 80 and lenght3 < 95 and lenght4 < 5:
                        cv2.putText(frame, "Angry Person", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    #Happy
                    elif lenght1 > 20 and lenght1 < 30 and lenght2 > 20 and lenght2 < 30 and lenght3 > 109 and lenght4 > 10 and lenght4 < 20:
                        cv2.putText(frame, "Happy Person", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                    #Surprise
                    elif lenght1 > 35 and lenght2 > 35 and lenght3 > 80 and lenght3 < 90 and lenght4 > 20:
                        cv2.putText(frame, "Surprised Person", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                    #Sad
                    elif lenght1 > 20 and lenght1 < 35 and lenght2 > 20 and lenght2 < 35 and lenght3 > 80 and lenght3 < 95 and lenght4 < 5:
                        cv2.putText(frame, "Sad Person", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)    
    cv2.imshow("Face Mesh Recognition", frame)
    t = cv2.waitKey(1)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
