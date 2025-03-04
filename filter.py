import os
import cv2
import numpy as np
from config import ORIGINAL_DIR, GAUSSIAN_DIR, SALT_PEPPER_DIR, DENOISED_DIR, MAX_IMAGES
from utils import *
import time
import cProfile

def apply_median_filter(image):
    return cv2.medianBlur(image, 5)

def apply_mean_filter(image): 
    # return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return cv2.blur(image, (5, 5))

def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def denoise_and_evaluate(img_array, original_img_array, name, save_to_disk=False):
    os.makedirs(DENOISED_DIR, exist_ok=True)
    os.makedirs(DENOISED_DIR + "/median", exist_ok=True)
    os.makedirs(DENOISED_DIR + "/mean", exist_ok=True)
    os.makedirs(DENOISED_DIR + "/gaussian", exist_ok=True)

    metrics = {
    'median': {'psnrs': [], 'ssims': []},
    'mean': {'psnrs': [], 'ssims': []},
    'gaussian': {'psnrs': [], 'ssims': []}
    }

    for i, noisy_image in enumerate(img_array):
        original = original_img_array[i]

        # Apply filters
        median_denoised = apply_median_filter(noisy_image)
        mean_denoised = apply_mean_filter(noisy_image)
        gaussian_denoised = apply_gaussian_filter(noisy_image)
        
        # Calculate PSNR and SSIM for each filter and store in the dictionary
        metrics['median']['psnrs'].append(calculate_psnr(original, median_denoised))
        metrics['median']['ssims'].append(calculate_ssim(original, median_denoised))

        metrics['mean']['psnrs'].append(calculate_psnr(original, mean_denoised))
        metrics['mean']['ssims'].append(calculate_ssim(original, mean_denoised))

        metrics['gaussian']['psnrs'].append(calculate_psnr(original, gaussian_denoised))
        metrics['gaussian']['ssims'].append(calculate_ssim(original, gaussian_denoised))
        
        if(save_to_disk):
            # Save images as PNGs
            cv2.imwrite(os.path.join(DENOISED_DIR + "/median", f"{name}_median_{i+1:04d}.png"), median_denoised)
            cv2.imwrite(os.path.join(DENOISED_DIR + "/mean", f"{name}_mean_{i+1:04d}.png"), mean_denoised)
            cv2.imwrite(os.path.join(DENOISED_DIR + "/gaussian", f"{name}_gaussian_{i+1:04d}.png"), gaussian_denoised)
        
        print(f"\rDenoising process ... {i+1} of {MAX_IMAGES}", end="", flush=True)

    return metrics

def denoise_and_evaluate_dataset(dataset, dataset_name):
    start = time.time()
    print("Denoising process ... ", end="")
    noisy_images, clean_images = load_images_from_dataset(dataset)
    metrics = denoise_and_evaluate(noisy_images, clean_images, dataset_name, save_to_disk=False)
    end = time.time()
    print(f"Total time elapsed: {(end - start):.4f} s")
    return metrics

def denoise_and_evaluate_default_folder():
    start = time.time()
    print("Denoising process ... ", end="")
    original_dataset = load_images_from_folder(ORIGINAL_DIR)
    gaussian_dataset = load_images_from_folder(GAUSSIAN_DIR)
    salt_pepper_dataset = load_images_from_folder(SALT_PEPPER_DIR)
    gaussian_metrics = denoise_and_evaluate(gaussian_dataset, original_dataset, "gaussian")
    print_metrics(gaussian_metrics, "gaussian")
    salt_pepper_metrics = denoise_and_evaluate(salt_pepper_dataset, original_dataset, "salt_pepper")
    print_metrics(salt_pepper_metrics, "salt_pepper")
    end = time.time()
    print(f"Total time elapsed: {(end - start):.4f} s")

if __name__ == "__main__":
    cProfile.run('denoise_and_evaluate_default_folder()', sort = 1)
