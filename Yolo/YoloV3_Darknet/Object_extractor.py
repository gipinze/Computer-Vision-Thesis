#This code should work for everyone who aims to extract the detected objects from video using YoloV3 Darknet
# It storages the detected objects in a designated folder creating unique folders for each class and runs in GPU, which is useful to visualize the video
# Also it detects and crops the object once every 0.5 seconds, this could be modified as well

# In this case detects Freezer, Goku and Vegeta from the videos, if you need the weights, leave a message


import cv2
import numpy as np
import os
import time
import colorsys

# Load YOLOv3 config and weights
net = cv2.dnn.readNetFromDarknet('cfg/yolov3-custom.cfg', 'backup/yolov3-custom_last.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# Define the classes and corresponding colors
classes = ['Freezer', 'Goku', 'Vegeta']
num_classes = len(classes)
hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
colors_by_class = dict(zip(classes, colors))

# Get the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers().tolist()]

# Load video file
cap = cv2.VideoCapture('data/DBZ-TEST/video1.mp4')

# Set the path to the output directory
output_dir = "data/DBZ-TEST/Extracted_faces"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create output folders for each class inside the output directory
for class_name in classes:
    class_folder = os.path.join(output_dir, class_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)


# Loop through each frame
frame_number = 0
last_saved_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects in the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

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
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # Add a check to make sure the bounding box is within the frame dimensions
                if x < 0:
                    w += x
                    x = 0
                if y < 0:
                    h += y
                    y = 0
                if x + w > frame.shape[1]:
                    w = frame.shape[1] - x
                if y + h > frame.shape[0]:
                    h = frame.shape[0] - y

                # Add margin to bounding box
                margin = 10
                x -= margin
                y -= margin
                w += margin*2
                h += margin*2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Draw bounding boxes around detected objects
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            if w <= 0 or h <= 0:
                continue
            margin = 10  # define margin size
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
            label = classes[class_ids[i]]
            color = colors_by_class[label]
            # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) # Hide the bounding box to not affect the cropped image
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save cropped objects to corresponding output folder once every 0.5 seconds
    current_time = time.time()
    if current_time - last_saved_time >= 0.5:
        for i in range(len(boxes)):
            if i in indexes:
                # Get the bounding box coordinates
                x, y, w, h = boxes[i]
                if w <= 0 or h <= 0:
                    continue
                margin = 10  # define margin size
                x1, y1 = max(0, x - margin), max(0, y - margin)
                x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)

                # Save image to corresponding output folder
                class_folder = os.path.join(output_dir, classes[class_ids[i]])
                output_path = os.path.join(class_folder, f"{frame_number}.jpg")
                cv2.imwrite(output_path, frame[y1:y2, x1:x2])



    # Show the processed frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()
