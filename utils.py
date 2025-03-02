import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity

def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)  # (BGR format)
        images.append(img)
    return images

def calculate_psnr(original, denoised):
    return cv2.PSNR(original, denoised)

def calculate_ssim(original, denoised):
    return structural_similarity(original, denoised, channel_axis=2, data_range=255)

def print_metrics(metrics, dataset_name=""):
    print(f"\n\n********** {dataset_name.capitalize()} Mean Metrics ***********")
    mean_metrics = {filter_name: {
        'mean_psnr': np.mean(metrics[filter_name]['psnrs']),
        'mean_ssim': np.mean(metrics[filter_name]['ssims'])
    } for filter_name in metrics}

    for filter_name, mean_metric in mean_metrics.items():
        print(f"{filter_name.capitalize()} (PSNR, SSIM): {mean_metric['mean_psnr']:.2f}, {mean_metric['mean_ssim']:.3f}")

    print("")
