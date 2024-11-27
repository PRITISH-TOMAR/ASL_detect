import os
import cv2
from Image_processing import func  # Ensure this custom module exists

# Define paths
raw_data_dir = "Data"  # Raw data directory containing folders A, B, C, ..., Z
output_dir = "data2"  # Output directory

# Create required directories for train and test splits
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Initialize counters
total_files_processed = 0
train_files_count = 0
test_files_count = 0

# Process each folder (A, B, C, ..., Z)
for folder_name in sorted(os.listdir(raw_data_dir)):
    folder_path = os.path.join(raw_data_dir, folder_name)
    
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder_name}")
        
        # Create corresponding subfolders in train and test directories
        train_subdir = os.path.join(train_dir, folder_name)
        test_subdir = os.path.join(test_dir, folder_name)
        os.makedirs(train_subdir, exist_ok=True)
        os.makedirs(test_subdir, exist_ok=True)

        # Get the list of files in the current folder
        files = os.listdir(folder_path)
        num_train_files = int(len(files) * 0.75)  # 75% for training
        num_test_files = len(files) - num_train_files  # Remaining for testing
        
        # Process and split files
        for idx, file_name in enumerate(files):
            file_path = os.path.join(folder_path, file_name)
            
            if os.path.isfile(file_path):
                total_files_processed += 1
                
                # Read and process the image
                img = cv2.imread(file_path, 0)  # Read as grayscale
                processed_img = func(file_path)  # Apply custom image processing
                
                # Save to train or test directories
                if idx < num_train_files:
                    train_files_count += 1
                    cv2.imwrite(os.path.join(train_subdir, file_name), processed_img)
                else:
                    test_files_count += 1
                    cv2.imwrite(os.path.join(test_subdir, file_name), processed_img)

# Print summary
print("Processing complete!")
print(f"Total files processed: {total_files_processed}")
print(f"Train files saved: {train_files_count}")
print(f"Test files saved: {test_files_count}")
