import os
import cv2
import time
import cProfile
import numpy as np
from config import ORIGINAL_DIR, GAUSSIAN_DIR, SALT_PEPPER_DIR, FOURIER_PATH, LOWPASS_CUTOFF, MAX_IMAGES, NO_PREVIEW
from utils import load_images_from_folder, calculate_psnr, calculate_ssim, print_metrics

def apply_fourier_lowpass_filter(image, cutoff=30):
    """
    Apply a Fourier low-pass filter to each color channel of the image using np.fft.
    :param image: Input image (BGR format).
    :param cutoff: Cutoff frequency for the low-pass filter.
    :return: The denoised image after applying the Fourier low-pass filter.
    """
    # Split the image into its BGR channels
    channels = cv2.split(image)
    
    # Apply Fourier low-pass filter to each channel
    filtered_channels = []
    for channel in channels:
        # Apply the Fast Fourier Transform (FFT)
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)  # Shift zero frequency to the center
        
        # Create a mask for low-pass filtering (zero out high-frequency components)
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        
        # Set the center region to pass the low frequencies and zero out high frequencies
        mask = np.zeros((rows, cols))
        mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
        
        # Apply the mask to the frequency domain representation
        fshift = fshift * mask
        
        # Inverse FFT to get the image back from the frequency domain
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        
        # Take the real part of the inverse FFT and normalize to range 0-255
        img_back = np.abs(img_back)
        img_back = np.uint8(np.clip(img_back, 0, 255))
        
        filtered_channels.append(img_back)
    
    # Merge the filtered channels back together
    denoised_image = cv2.merge(filtered_channels)
    
    return denoised_image

def denoise_and_evaluate(dataset, original_dataset, dataset_name):
    os.makedirs(FOURIER_PATH, exist_ok=True)

    # ******************************************************** TEMP ***************************************
    first_comparison_done = NO_PREVIEW
    # ******************************************************** TEMP ***************************************

    metrics = {
    'fourier': {'psnrs': [], 'ssims': []}
    }

    for i, noisy_image in enumerate(dataset):
        original = original_dataset[i]

        # Apply Fourier low-pass filter
        fourier_denoised = apply_fourier_lowpass_filter(noisy_image, cutoff=LOWPASS_CUTOFF)
        
        # Calculate PSNR and SSIM
        fourier_psnr = calculate_psnr(original, fourier_denoised)
        fourier_ssim = calculate_ssim(original, fourier_denoised)

         # Calculate PSNR and SSIM and store in the dictionary
        metrics['fourier']['psnrs'].append(fourier_psnr)
        metrics['fourier']['ssims'].append(fourier_ssim)
        
        # Save results as PNG images
        cv2.imwrite(os.path.join(FOURIER_PATH, f"{dataset_name}_fourier_{i+1:04d}.png"), fourier_denoised)

        # ******************************************************** TEMP ***************************************
        if not first_comparison_done:
            combined_image = np.hstack((original, noisy_image))
            cv2.imshow("Original vs Noisy Image", combined_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            first_comparison_done = True
        # ******************************************************** TEMP ***************************************
        
        print(f"\rFourier denoising process ... {i+1} of {MAX_IMAGES}", end="", flush=True)

    print_metrics(metrics, dataset_name)

def main():
    start = time.time()
    print("Fourier denoising process ...", end="")
    original_dataset = load_images_from_folder(ORIGINAL_DIR)
    gaussian_dataset = load_images_from_folder(GAUSSIAN_DIR)
    salt_pepper_dataset = load_images_from_folder(SALT_PEPPER_DIR)
    denoise_and_evaluate(gaussian_dataset, original_dataset, "gaussian")
    denoise_and_evaluate(salt_pepper_dataset, original_dataset, "salt_pepper")
    end = time.time()
    print(f"Total time elapsed: {(end - start):.4f} s")

if __name__ == "__main__":
    cProfile.run('main()', sort = 1)