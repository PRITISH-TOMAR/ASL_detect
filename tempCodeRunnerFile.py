import os
import cv2

# Directory to save the dataset
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes and images per class
number_of_classes = 3
dataset_size = 100

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Loop through each class
for class_id in range(number_of_classes):
    # Create a directory for the class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, chr( 65 + class_id))
    if not os.path.exists(class_dir):
