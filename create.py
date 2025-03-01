import numpy as np
from skimage.util import random_noise
from math import sqrt
import scipy.io
import os
import datetime
import cv2  # OpenCV for saving images as PNG

# Global variables
data_path = "./data/flowers-102/jpg"
mat_file_path = "./data/flowers-102/imagelabels.mat"
split_file_path = "./data/flowers-102/setid.mat"
original_path = "./results/original"
gaussian_path = "./results/gaussian"
salt_pepper_path = "./results/salt_pepper"
image_size = (128, 128)

def load_matlab_splits():
    """
    Load MATLAB files containing dataset splits and labels.
    """
    labels = scipy.io.loadmat(mat_file_path)["labels"].flatten()
    splits = scipy.io.loadmat(split_file_path)
    train = splits["trnid"].flatten()
    val = splits["valid"].flatten()
    
    selected_indices = np.concatenate((train, val)) - 1  # Convert to 0-based index
    return selected_indices, labels

def load_and_preprocess_image(image_path):
    # Use OpenCV to load the image (which will be in BGR format)
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, image_size)
    return image_resized

def add_gaussian_noise(image, std):
    noise = np.random.normal(0, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def add_salt_and_pepper_noise(image, ratio):
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    noisy_pixels = int(h * w * ratio)
 
    for _ in range(noisy_pixels):
        row, col = np.random.randint(0, h), np.random.randint(0, w)
        if np.random.rand() < 0.5:
            noisy_image[row, col] = [0, 0, 0] 
        else:
            noisy_image[row, col] = [255, 255, 255]
 
    return noisy_image

def create_noisy_datasets():
    """
    Create datasets with Gaussian and salt & pepper noise, selecting 2040 images based on splits.
    """
    selected_indices, _ = load_matlab_splits()
    
    # Ensure the directories for saving images exist
    os.makedirs(original_path, exist_ok=True)
    os.makedirs(gaussian_path, exist_ok=True)
    os.makedirs(salt_pepper_path, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(data_path) if f.endswith(('.jpg'))])

    print(f"[{datetime.datetime.now()}] Loading complete. Processing images ...")
    
    for i, _ in enumerate(selected_indices):
        if(i % 100 == 0):
            print(f"[{datetime.datetime.now()}] {i} of 2040")
                  
        image_path = os.path.join(data_path, image_files[i])
        image = load_and_preprocess_image(image_path)
        
        # Save original image as PNG
        original_filename = os.path.join(original_path, f"image_{i+1:04d}.png")
        cv2.imwrite(original_filename, image)
        
        # Add noise and save noisy images
        gaussian_noisy_image = add_gaussian_noise(image, sqrt(0.5) * 255)
        gaussian_filename = os.path.join(gaussian_path, f"image_{i+1:04d}_gaussian.png")
        cv2.imwrite(gaussian_filename, gaussian_noisy_image)
        
        salt_pepper_noisy_image = add_salt_and_pepper_noise(image, 0.5)
        salt_pepper_filename = os.path.join(salt_pepper_path, f"image_{i+1:04d}_salt_pepper.png")
        cv2.imwrite(salt_pepper_filename, salt_pepper_noisy_image)
        
    print(f"[{datetime.datetime.now()}] Created and saved all 2040 PNGs.")

if __name__ == "__main__":
    create_noisy_datasets()
