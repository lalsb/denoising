import os
import cv2 
import scipy.io
import time
import cProfile
import numpy as np
from math import sqrt
from torchvision.datasets import Flowers102
from skimage.util import random_noise, img_as_float, img_as_ubyte, dtype_limits
from skimage.exposure import rescale_intensity
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

def add_gaussian_noise(image, var):
    float_image = img_as_float(image)
    noised_image = random_noise(float_image, mode='gaussian', var=var,  clip=False)
    normalized_image = rescale_intensity(noised_image, in_range = 'image', out_range = (0,1))
    ubyte_image = img_as_ubyte(normalized_image)
    return ubyte_image

def add_salt_and_pepper_noise(image, amount):
    image = img_as_float(image)
    image = random_noise(image, mode='s&p', amount=amount, clip=True)
    return img_as_ubyte(image)

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

    #gaussian_files = []
    #salt_pepper_files = []

    for i, _ in enumerate(selected_indices):
        print(f"\rNoising process ... {i+1} of {MAX_IMAGES}", end="", flush=True)
                  
        image_path = os.path.join(DATA_DIR, image_files[i])
        image = load_and_preprocess_image(image_path)
        
        # Save orignal images as PNGs
        original_filename = os.path.join(ORIGINAL_DIR, f"image_{i+1:04d}.png")
        cv2.imwrite(original_filename, image)
        
        # Apply gaussian noise and save images as PNGs
        gaussian_noisy_image = add_gaussian_noise(image, 0.5)
        gaussian_filename = os.path.join(GAUSSIAN_DIR, f"image_{i+1:04d}_gaussian.png")
        cv2.imwrite(gaussian_filename, gaussian_noisy_image)
        #gaussian_files.append(gaussian_noisy_image)
        
        # Apply salt and pepper noise and save images as PNGs
        salt_pepper_noisy_image = add_salt_and_pepper_noise(image, 0.5)
        salt_pepper_filename = os.path.join(SALT_PEPPER_DIR, f"image_{i+1:04d}_salt_pepper.png")
        cv2.imwrite(salt_pepper_filename, salt_pepper_noisy_image)
        #salt_pepper_files.append(salt_pepper_noisy_image)

        # Calculate PSNR and SSIM for each filter and store in the dictionary
        metrics['gaussian']['psnrs'].append(calculate_psnr(image, gaussian_noisy_image))
        metrics['gaussian']['ssims'].append(calculate_ssim(image, gaussian_noisy_image))

        metrics['salt_pepper']['psnrs'].append(calculate_psnr(image, salt_pepper_noisy_image))
        metrics['salt_pepper']['ssims'].append(calculate_ssim(image, salt_pepper_noisy_image))
    
    #np.savez(f"{GAUSSIAN_DIR}/gaussian_files.npz", images = gaussian_files)
    #np.savez(f"{SALT_PEPPER_DIR}/salt_pepper_files.npz", images = salt_pepper_files)
    print_metrics(metrics, "Mean")

def main():
    start = time.time()
    print(f"Noising process ... ", end="")
    Flowers102(root="./data", split="train", download=True)
    create_noisy_datasets()
    end = time.time()
    print(f"Total time elapsed: {(end - start):.4f} s")

if __name__ == "__main__":
    cProfile.run('main()', sort = 1)
