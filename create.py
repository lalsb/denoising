import numpy as np
from math import sqrt
import scipy.io
import os
import cv2 
from config import DATA_DIR, SPLIT_FILE, ORIGINAL_DIR, GAUSSIAN_DIR, SALT_PEPPER_DIR, IMAGE_SIZE, MAX_IMAGES
from utils import calculate_psnr, calculate_ssim, print_metrics

def load_matlab_splits():
    splits = scipy.io.loadmat(SPLIT_FILE)
    train = splits["trnid"].flatten()
    val = splits["valid"].flatten()
    
    selected_indices = np.concatenate((train, val)) - 1  # to 0-based index
    selected_indices = selected_indices[:MAX_IMAGES]
    return selected_indices

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, IMAGE_SIZE)
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
    Create datasets with Gaussian and salt & pepper noise, selecting {MAX_IMAGES} images based on splits.
    """
    selected_indices = load_matlab_splits()

    metrics = {
    'gaussian': {'psnrs': [], 'ssims': []},
    'salt_pepper': {'psnrs': [], 'ssims': []}
    }
    
    os.makedirs(ORIGINAL_DIR, exist_ok=True)
    os.makedirs(GAUSSIAN_DIR, exist_ok=True)
    os.makedirs(SALT_PEPPER_DIR, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(('.jpg'))])

    for i, _ in enumerate(selected_indices):
        print(f"\rNoising process ... {i+1} of {MAX_IMAGES}", end="", flush=True)
                  
        image_path = os.path.join(DATA_DIR, image_files[i])
        image = load_and_preprocess_image(image_path)
        
        # Save orignal images as PNGs
        original_filename = os.path.join(ORIGINAL_DIR, f"image_{i+1:04d}.png")
        cv2.imwrite(original_filename, image)
        
        # Apply gaussian noise and save images as PNGs
        gaussian_noisy_image = add_gaussian_noise(image, sqrt(0.5) * 255)
        gaussian_filename = os.path.join(GAUSSIAN_DIR, f"image_{i+1:04d}_gaussian.png")
        cv2.imwrite(gaussian_filename, gaussian_noisy_image)
        
        # Apply salt and pepper noise and save images as PNGs
        salt_pepper_noisy_image = add_salt_and_pepper_noise(image, 0.5)
        salt_pepper_filename = os.path.join(SALT_PEPPER_DIR, f"image_{i+1:04d}_salt_pepper.png")
        cv2.imwrite(salt_pepper_filename, salt_pepper_noisy_image)

        # Calculate PSNR and SSIM for each filter and store in the dictionary
        metrics['gaussian']['psnrs'].append(calculate_psnr(image, gaussian_noisy_image))
        metrics['gaussian']['ssims'].append(calculate_ssim(image, gaussian_noisy_image))

        metrics['salt_pepper']['psnrs'].append(calculate_psnr(image, salt_pepper_noisy_image))
        metrics['salt_pepper']['ssims'].append(calculate_ssim(image, salt_pepper_noisy_image))
    
    print_metrics(metrics, "Mean")

if __name__ == "__main__":
    print(f"Noising process ... ", end="")
    create_noisy_datasets()
