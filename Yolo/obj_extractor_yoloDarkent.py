import cv2
import os
import numpy as np


cfg = r'../Darknet/cfg/yolov3-custom.cfg'
weights = r'../Darknet/backup/yolov3-custom_last.weights'
data = '../Darknet/data/obj.names'


# Load YOLOv3 from Darknet
net = cv2.dnn.readNetFromDarknet(cfg, weights )
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# define the labels
classes = []

with open(data, "r") as f:
    classes = [line.strip() for line in f.readlines()]
# Get the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers().tolist()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Path to folder with images
folder_path = "img_test"

# Path to folder for output
output_folder_path = "output"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Loop through images in folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Load image
        img = cv2.imread(os.path.join(folder_path, filename))

        # Get image dimensions
        height, width, channels = img.shape

        # Detect objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Loop through detected objects
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    label = str(classes[class_id])
                    color = colors[class_id]
                    x, y, w, h = detection[:4] * np.array([width, height, width, height])
                    x = int(x - w / 2)
                    y = int(y - h / 2)
                    w = int(w)
                    h = int(h)
                    
                    # Save image of detected object in corresponding folder
                    output_folder = os.path.join(output_folder_path, label)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    output_filename = os.path.splitext(filename)[0] + "_" + label + os.path.splitext(filename)[1]
                    cropped_img = img[y:y+h, x:x+w]
                    if cropped_img.shape[0] and cropped_img.shape[1]:
                        cv2.imwrite(os.path.join(output_folder, output_filename), cropped_img)

print("Iteration done")
