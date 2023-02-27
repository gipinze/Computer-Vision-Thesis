""" Python code to extract faces using MTCNN from VGGFace2"""

import os
from cv2 import cv2
from mtcnn import MTCNN

# Define the MTCNN detector
detector = MTCNN()

# Define the output size of the cropped faces
OUTPUT_SIZE = 256

# Define the output directories to save the cropped faces
train_output_dir = 'W:/10_percent_test/train_crop'
test_output_dir = 'W:/10_percent_test/test_crop'


# Define the train and test folders
train_dir = 'W:/10_percent_test/train/'
test_dir = 'W:/10_percent_test/test/'

# Loop through each image in the train folder and detect and crop the faces
for root, dirs, files in os.walk(train_dir):
    for file in files:
        # Get the input image path
        input_path = os.path.join(root, file)
        # Load the input image
        img = cv2.imread(input_path)
        # Detect the faces using MTCNN
        faces = detector.detect_faces(img)
        # Loop through each detected face and crop it
        for face in faces:
            x, y, w, h = face['box']
            # Add 10 pixels to the bounding box
            x -= 10
            y -= 10
            w += 20
            h += 20
            # Crop the face
            cropped_face = img[y:y+h, x:x+w]
            # Check if the cropped face size is valid
            if cropped_face.size != 0:
                # Resize the face to the output size
                cropped_face = cv2.resize(cropped_face, (OUTPUT_SIZE, OUTPUT_SIZE))
                # Get the output directory for the cropped face
                output_subdir = os.path.join(train_output_dir, os.path.relpath(root, train_dir))
                # Create the output directory if it does not exist
                os.makedirs(output_subdir, exist_ok=True)
                # Get the output path for the cropped face
                output_path = os.path.join(output_subdir, file)
                # Save the cropped face to the output path
                cv2.imwrite(output_path, cropped_face)

# Loop through each image in the test folder and detect and crop the faces
for root, dirs, files in os.walk(test_dir):
    for file in files:
        # Get the input image path
        input_path = os.path.join(root, file)
        # Load the input image
        img = cv2.imread(input_path)
        # Detect the faces using MTCNN
        faces = detector.detect_faces(img)
        # Loop through each detected face and crop it
        for face in faces:
            x, y, w, h = face['box']
            # Add 10 pixels to the bounding box
            x -= 10
            y -= 10
            w += 20
            h += 20
            # Crop the face
            cropped_face = img[y:y+h, x:x+w]
            # Check if the cropped face size is valid
            if cropped_face.size != 0:
                # Resize the face to the output size
                cropped_face = cv2.resize(cropped_face, (OUTPUT_SIZE, OUTPUT_SIZE))
                # Get the output directory for the cropped face
                output_subdir = os.path.join(test_output_dir, os.path.relpath(root, test_dir))
                # Create the output directory if it does not exist
                os.makedirs(output_subdir, exist_ok=True)
                # Get the output path for the cropped face
                output_path = os.path.join(output_subdir, file)
                # Save the cropped face to the output path
                cv2.imwrite(output_path, cropped_face)
