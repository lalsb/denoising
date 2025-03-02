import os
import cv2
import numpy as np
from config import ORIGINAL_DIR, GAUSSIAN_DIR, SALT_PEPPER_DIR, DENOISED_DIR, MAX_IMAGES, NO_PREVIEW
from utils import load_images_from_folder, calculate_psnr, calculate_ssim, print_metrics
import time
import cProfile

def apply_median_filter(image):
    return cv2.medianBlur(image, 5)

def apply_mean_filter(image): 
    # return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return cv2.blur(image, (5, 5))

def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def denoise_and_evaluate(dataset, original_dataset, dataset_name):
    os.makedirs(DENOISED_DIR, exist_ok=True)
    os.makedirs(DENOISED_DIR + "/median", exist_ok=True)
    os.makedirs(DENOISED_DIR + "/mean", exist_ok=True)
    os.makedirs(DENOISED_DIR + "/gaussian", exist_ok=True)

    # ******************************************************** TEMP ***************************************
    first_comparison_done = NO_PREVIEW
    # ******************************************************** TEMP ***************************************

    metrics = {
    'median': {'psnrs': [], 'ssims': []},
    'mean': {'psnrs': [], 'ssims': []},
    'gaussian': {'psnrs': [], 'ssims': []}
    }

    for i, noisy_image in enumerate(dataset):
        original = original_dataset[i]


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
        
        # Save images as PNGs
        cv2.imwrite(os.path.join(DENOISED_DIR + "/median", f"{dataset_name}_median_{i+1:04d}.png"), median_denoised)
        cv2.imwrite(os.path.join(DENOISED_DIR + "/mean", f"{dataset_name}_mean_{i+1:04d}.png"), mean_denoised)
        cv2.imwrite(os.path.join(DENOISED_DIR + "/gaussian", f"{dataset_name}_gaussian_{i+1:04d}.png"), gaussian_denoised)

        # ******************************************************** TEMP ***************************************
        if not first_comparison_done:
            combined_image = np.hstack((original, noisy_image, median_denoised, mean_denoised, gaussian_denoised))
            cv2.imshow("Original vs Noisy vs Filtered Image", combined_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            first_comparison_done = True
        # ******************************************************** TEMP ***************************************
        
        print(f"\rDenoising process ... {i+1} of {MAX_IMAGES}", end="", flush=True)

    print_metrics(metrics, dataset_name)

def main():
    start = time.time()
    print("Denoising process ... ", end="")
    original_dataset = load_images_from_folder(ORIGINAL_DIR)
    gaussian_dataset = load_images_from_folder(GAUSSIAN_DIR)
    salt_pepper_dataset = load_images_from_folder(SALT_PEPPER_DIR)
    denoise_and_evaluate(gaussian_dataset, original_dataset, "gaussian")
    denoise_and_evaluate(salt_pepper_dataset, original_dataset, "salt_pepper")
    end = time.time()
    print(f"Total time elapsed: {(end - start):.4f} s")

if __name__ == "__main__":
    cProfile.run('main()', sort = 1)
