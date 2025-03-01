import os
import datetime
import cv2
import numpy as np
from skimage.metrics import structural_similarity

# Global variables
original_path = "./results/original"
gaussian_path = "./results/gaussian"
salt_pepper_path = "./results/salt_pepper"
fourier_path = "./results/denoised/fourier"
image_size = (128, 128)
lowpass_cutoff = 32 

print(f"[{datetime.datetime.now()}] Starting Fourier denoising process...")

def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)  # Read image in BGR format
            if img is not None:
                images.append(img)
    return images

def fourier_lowpass_filter(image, cutoff=30):
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

def calculate_psnr(original, denoised):
    return cv2.PSNR(original, denoised)

def calculate_ssim(original, denoised):
    return structural_similarity(original, denoised, channel_axis=2, data_range=255)

def denoise_and_evaluate(dataset, original_dataset, dataset_name):
    os.makedirs(fourier_path, exist_ok=True)
    
    print(f"[{datetime.datetime.now()}] Loading complete. Processing images ...")

    # Show the first pair of original and noisy images
    first_comparison_done = False

    for i, noisy_image in enumerate(dataset):
        original = original_dataset[i]

        # Apply Fourier low-pass filter
        fourier_denoised = fourier_lowpass_filter(noisy_image, cutoff=lowpass_cutoff)
        
        # Calculate PSNR and SSIM
        fourier_psnr = calculate_psnr(original, fourier_denoised)
        fourier_ssim = calculate_ssim(original, fourier_denoised)
        
        # Save results as PNG images
        cv2.imwrite(os.path.join(fourier_path, f"{dataset_name}_fourier_{i+1:04d}.png"), fourier_denoised)

        if not first_comparison_done:
            # Show comparison of the first pair of original and noisy images
            combined_image = np.hstack((original, noisy_image))  # Stack images horizontally
            cv2.imshow("Original vs Noisy Image", combined_image)
            cv2.waitKey(0)  # Wait until the user presses a key
            cv2.destroyAllWindows()  # Close the window
            first_comparison_done = True
        
        
        if i % 100 == 0:
            print(f"[{datetime.datetime.now()}] {i} of 2040 - (PSNR, SSIM) ({fourier_psnr:.2f}, {fourier_ssim:.3f})")

if __name__ == "__main__":
    # Load datasets from directories (original, gaussian, and salt_pepper images)
    original_dataset = load_images_from_folder(original_path)
    gaussian_dataset = load_images_from_folder(gaussian_path)
    salt_pepper_dataset = load_images_from_folder(salt_pepper_path)
    
    # Denoise and evaluate both datasets
    denoise_and_evaluate(gaussian_dataset, original_dataset, "gaussian")
    denoise_and_evaluate(salt_pepper_dataset, original_dataset, "salt_pepper")
    
    print(f"[{datetime.datetime.now()}] Fourier denoising complete.")
