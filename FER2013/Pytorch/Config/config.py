
# import the necessary packages
import os

# initialize the path to the root folder where the dataset resides and the
# path to the train and test directory
DATASET_FOLDER = f'T:/Thesis project/Python Projects/Thesis Files/FER2013/FER2013_Dataset'
trainDirectory = os.path.join(DATASET_FOLDER, "train")
testDirectory = os.path.join(DATASET_FOLDER, "test")

# initialize the amount of samples to use for training and validation
TRAIN_SIZE = 0.90
VAL_SIZE = 0.10

# specify the batch size, total number of epochs and the learning rate
BATCH_SIZE = 64 # This previous number was 16, and then 32 but, until further notice, 32 seems like a better option (is the standard)
NUM_OF_EPOCHS = 100
LR = 1e-1