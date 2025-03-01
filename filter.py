import os
import datetime
import cv2  # OpenCV for loading, processing, and saving images
from skimage.metrics import structural_similarity
import numpy as np

# Global variables
original_path = "./results/original"
gaussian_path = "./results/gaussian"
salt_pepper_path = "./results/salt_pepper"
denoised_path = "./results/denoised"
image_size = (128, 128)

print(f"[{datetime.datetime.now()}] Starting denoising process...")

def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)  # Read image in BGR format
            if img is not None:
                images.append(img)
    return images

def apply_median_filter(image):
    # OpenCV's median blur filter
    return cv2.medianBlur(image, 5)

def apply_mean_filter(image): 
    # OpenCV's non-local means denoising (for color images)
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def apply_gaussian_filter(image):
    # OpenCV's Gaussian blur
    return cv2.GaussianBlur(image, (5, 5), 0)

def calculate_psnr(original, denoised):
    return cv2.PSNR(original, denoised)

def calculate_ssim(original, denoised):
    return structural_similarity(original, denoised, channel_axis=2, data_range=255)

def denoise_and_evaluate(dataset, original_dataset, dataset_name):
    os.makedirs(denoised_path, exist_ok=True)
    os.makedirs(denoised_path + "/median", exist_ok=True)
    os.makedirs(denoised_path + "/mean", exist_ok=True)
    os.makedirs(denoised_path + "/gaussian", exist_ok=True)
    
    print(f"[{datetime.datetime.now()}] Loading complete. Processing images ...")

    # Show the first pair of original and noisy images
    first_comparison_done = False

    for i, noisy_image in enumerate(dataset):
        original = original_dataset[i]

        # Apply filters
        median_denoised = apply_median_filter(noisy_image)
        mean_denoised = apply_mean_filter(noisy_image)
        gaussian_denoised = apply_gaussian_filter(noisy_image)
        
        # Calculate PSNR and SSIM
        median_psnr = calculate_psnr(original, median_denoised)
        median_ssim = calculate_ssim(original, median_denoised)
        
        mean_psnr = calculate_psnr(original, mean_denoised)
        mean_ssim = calculate_ssim(original, mean_denoised)
        
        gaussian_psnr = calculate_psnr(original, gaussian_denoised)
        gaussian_ssim = calculate_ssim(original, gaussian_denoised)
        
        # Save results as PNG images
        cv2.imwrite(os.path.join(denoised_path + "/median", f"{dataset_name}_median_{i+1:04d}.png"), median_denoised)
        cv2.imwrite(os.path.join(denoised_path + "/mean", f"{dataset_name}_mean_{i+1:04d}.png"), mean_denoised)
        cv2.imwrite(os.path.join(denoised_path + "/gaussian", f"{dataset_name}_gaussian_{i+1:04d}.png"), gaussian_denoised)

        if not first_comparison_done:
            # Show comparison of the first pair of original and noisy images
            combined_image = np.hstack((original, noisy_image, median_denoised, mean_denoised, gaussian_denoised))  # Stack images horizontally
            cv2.imshow("Original vs Noisy vs Filtered Image", combined_image)
            cv2.waitKey(0)  # Wait until the user presses a key
            cv2.destroyAllWindows()  # Close the window
            first_comparison_done = True
        
        if(i % 100 == 0):
            print(f"[{datetime.datetime.now()}] {i} of 2040 - (PSNR, SSIM): Median ({median_psnr:.2f}, {median_ssim:.3f}), "
                  f"Mean ({mean_psnr:.2f}, {mean_ssim:.3f}), Gaussian ({gaussian_psnr:.2f}, {gaussian_ssim:.3f})")

if __name__ == "__main__":
    # Load datasets from directories (original, gaussian, and salt_pepper images)
    original_dataset = load_images_from_folder(original_path)
    gaussian_dataset = load_images_from_folder(gaussian_path)
    salt_pepper_dataset = load_images_from_folder(salt_pepper_path)
    
    # Denoise and evaluate both datasets
    denoise_and_evaluate(gaussian_dataset, original_dataset, "gaussian")
    denoise_and_evaluate(salt_pepper_dataset, original_dataset, "salt_pepper")
    
    print(f"[{datetime.datetime.now()}] Denoising complete.")
