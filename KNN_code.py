import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define paths for training and testing data
train_path = "Kaggle Dataset for Brain Tumour Detection MRI/Training"
test_path = "Kaggle Dataset for Brain Tumour Detection MRI/Testing"

# Classes in the dataset
categories = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# Image size for resizing
IMG_SIZE = 128

# Function to load and preprocess images
def load_images_from_folder(folder_path, categories, img_size):
    data = []
    labels = []
    for category in categories:
        class_path = os.path.join(folder_path, category)
        label = categories.index(category)  # Assign a numerical label for each class
        for img_file in os.listdir(class_path):
            try:
                # Load the image in grayscale
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                # Resize the image to the specified size
                img = cv2.resize(img, (img_size, img_size))
                # Append to data and labels
                data.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_file}: {e}")
    return np.array(data), np.array(labels)

# Load training and testing data
train_data, train_labels = load_images_from_folder(train_path, categories, IMG_SIZE)
test_data, test_labels = load_images_from_folder(test_path, categories, IMG_SIZE)

# Normalize the image data
train_data = train_data / 255.0
test_data = test_data / 255.0

# Flatten the images for k-NN
train_data = train_data.reshape(len(train_data), -1)  # Convert to 1D feature vectors
test_data = test_data.reshape(len(test_data), -1)

# Print the shape of the datasets
print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")
