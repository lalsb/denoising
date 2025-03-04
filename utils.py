from torch.utils.data import DataLoader
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity

def load_images_from_folder(path_to_folder):
    images = []
    for filename in sorted(os.listdir(path_to_folder)):
        img_path = os.path.join(path_to_folder, filename)
        img = cv2.imread(img_path)  # (BGR format)
        images.append(img)
    return images

def numpy_collate(batch):
    return batch
def load_images_from_dataset(dataset):
    dataloader = DataLoader(dataset, batch_size=None, shuffle=True, collate_fn=numpy_collate)
    noisy_images, clean_images = [], []

    for noisy_image, clean_image in dataloader:
        noisy_images.append(noisy_image)
        clean_images.append(clean_image)

    return noisy_images, clean_images 

def calculate_metrics_from_dataset(dataset):
    metrics = {
    'original': {'psnrs': [], 'ssims': []}
    }
    noisy_images, clean_images = load_images_from_dataset(dataset)
    for noisy_image, clean_image in zip(noisy_images, clean_images):
        psnr = calculate_psnr(noisy_image, clean_image)
        ssim = calculate_ssim(noisy_image, clean_image)
        metrics['original']['psnrs'].append(psnr)
        metrics['original']['ssims'].append(ssim)
    return metrics

def calculate_psnr(original, denoised):
    return cv2.PSNR(original, denoised)

def calculate_ssim(original, denoised):
    return structural_similarity(original, denoised, channel_axis=2, data_range=255)

def print_metrics(metrics, dataset_name=""):
    mean_metrics = {filter_name: {
        'mean_psnr': np.mean(metrics[filter_name]['psnrs']),
        'mean_ssim': np.mean(metrics[filter_name]['ssims'])
    } for filter_name in metrics}
    print(f"\r", end="", flush=True)
    for filter_name, mean_metric in mean_metrics.items():
        print(f"{filter_name.capitalize()} (PSNR, SSIM): {mean_metric['mean_psnr']:.2f} dB, {mean_metric['mean_ssim']:.3f}")

