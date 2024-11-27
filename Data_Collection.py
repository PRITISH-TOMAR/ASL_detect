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
        os.makedirs(class_dir)

    print(f'Collecting data for class { chr(65 + class_id)}')

    # Wait for the user to start data collection
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.putText(
            frame,
            'To begin capture, press "A".',
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 255),
            3,
            cv2.LINE_AA
        )
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('a'):
            break

    # Capture images for the class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

# Release resources
cap.release()
cv2.destroyAllWindows()