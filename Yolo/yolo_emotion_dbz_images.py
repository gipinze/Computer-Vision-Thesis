import cv2
import numpy as np
from keras.models import load_model

# Load YOLOv3 config and weights
net = cv2.dnn.readNetFromDarknet('cfg/yolov3-custom.cfg', 'backup/yolov3-custom_last.weights')

# Load emotion recognition model
model_h5 = r"C:/Users/darks/Downloads/MobileNetV2_tf_448_rgb50+30.h5"
emotion_model = load_model(model_h5)

# Define the classes and corresponding colors
classes = ['Anger', 'Happy', 'Sad', 'Surprise']
num_classes = len(classes)
colors = np.random.uniform(0, 255, size=(num_classes, 4))

# Load image file
image_file = 'path/to/img'
image = cv2.imread(image_file)

# Detect objects in the image
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward(net.getUnconnectedOutLayersNames())

# Filter out low confidence detections
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Process each face detected
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        if w <= 0 or h <= 0:
            continue
        # Extract face region
        face = image[y:y+h, x:x+w]
        # Check if face is not empty
        if face.size == 0:
            continue
        # Resize and preprocess image for emotion recognition model
        face = cv2.resize(face, (448, 448))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)
        face = face / 255.0

        # Predict emotion using the model
        emotions = emotion_model.predict(face)
        emotion_label = classes[np.argmax(emotions)]

        # Draw bounding boxes and emotion labels
        label = "{}: {:.2f}".format(emotion_label, np.max(emotions))
        color = colors[class_ids[i]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the image with bounding boxes and emotion labels
cv2.imshow('Object detection and Emotion recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()