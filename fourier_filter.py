import os
import cv2
import time
import cProfile
import numpy as np
from config import ORIGINAL_DIR, GAUSSIAN_DIR, SALT_PEPPER_DIR, FOURIER_PATH, LOWPASS_CUTOFF, MAX_IMAGES
from utils import *

def apply_fourier_lowpass_filter(image, radius=20):

    h, w, _ = image.shape
    filtered_channels = []
    channels = cv2.split(image)

    for channel in channels:  # Process each BGR channel separately

        # Perform FFT and shift zero frequency to center
        dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shifted = np.fft.fftshift(dft)

        # Create a circular mask
        mask = np.zeros((h, w, 2), np.uint8)
        cy = mask.shape[0] // 2
        cx = mask.shape[1] // 2
        cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]

        # Apply mask and inverse transform
        dft_filtered = dft_shifted * mask
        dft_ishifted = np.fft.ifftshift(dft_filtered)
        img_back = cv2.idft(dft_ishifted)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Normalize result to 0-255
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        filtered_channels.append(img_back.astype(np.uint8))

    # Merge channels back into a BGR image
    filtered_image = cv2.merge(filtered_channels)
    
    return filtered_image


def denoise_and_evaluate(dataset, original_dataset, dataset_name="", save_to_disk=False):
    os.makedirs(FOURIER_PATH, exist_ok=True)

    metrics = {
    'fourier': {'psnrs': [], 'ssims': []}
    }

    for i, noisy_image in enumerate(dataset):
        original = original_dataset[i]

        # Apply Fourier low-pass filter
        fourier_denoised = apply_fourier_lowpass_filter(noisy_image)
        
        # Calculate PSNR and SSIM
        fourier_psnr = calculate_psnr(original, fourier_denoised)
        fourier_ssim = calculate_ssim(original, fourier_denoised)

        # Calculate PSNR and SSIM and store in the dictionary
        metrics['fourier']['psnrs'].append(fourier_psnr)
        metrics['fourier']['ssims'].append(fourier_ssim)
        
        if(save_to_disk):
            # Save results as PNGs
            cv2.imwrite(os.path.join(FOURIER_PATH, f"{dataset_name}_fourier_{i+1:04d}.png"), fourier_denoised)

        print(f"\rDenoising process ... {i+1} of {MAX_IMAGES}", end="", flush=True)

    return metrics

def denoise_and_evaluate_dataset(dataset):
     noisy_images, clean_images = load_images_from_dataset(dataset)
     metrics = denoise_and_evaluate(noisy_images, clean_images, save_to_disk=False)
     return metrics

def denoise_and_evaluate_default_folder():
    original_dataset = load_images_from_folder(ORIGINAL_DIR)
    gaussian_dataset = load_images_from_folder(GAUSSIAN_DIR)
    salt_pepper_dataset = load_images_from_folder(SALT_PEPPER_DIR)
    gaussian_metrics = denoise_and_evaluate(gaussian_dataset, original_dataset, "gaussian", save_to_disk=True)
    print_metrics(gaussian_metrics, "gaussian")
    salt_pepper_metrics = denoise_and_evaluate(salt_pepper_dataset, original_dataset, "salt_pepper", save_to_disk=True)
    print_metrics(salt_pepper_metrics, "salt_pepper")

if __name__ == "__main__":
    cProfile.run('denoise_and_evaluate_default_folder()', sort = 1)