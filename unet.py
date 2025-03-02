import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, random_split
from skimage.util import random_noise, img_as_float, img_as_ubyte
from config import ORIGINAL_DIR, GAUSSIAN_DIR, SALT_PEPPER_DIR, FOURIER_PATH, LOWPASS_CUTOFF, MAX_IMAGES, NO_PREVIEW
from utils import load_images_from_folder, calculate_psnr, calculate_ssim, print_metrics

# Global vars
TRAIN_EPOCHS = 100
TEST_EPOCHS = 36
BATCH_SIZE = None
TRAIN_RATIO = 0.8

# Dataset (noisy image, original image)
class DenoisingDataset(Dataset):
    def __init__(self, noisy_images, clean_images):
        self.noisy_images = noisy_images
        self.clean_images = clean_images
    
    def __len__(self):
        return len(self.noisy_images)
    
    def __getitem__(self, i):
        return self.noisy_images[i], self.clean_images[i]

# Load image arrays
original_img_array = load_images_from_folder(ORIGINAL_DIR)
gaussian_img_array = load_images_from_folder(GAUSSIAN_DIR)
salt_pepper_img_array = load_images_from_folder(SALT_PEPPER_DIR)

# Create datasets
gaussian_datasets = DenoisingDataset(gaussian_img_array, original_img_array)
salt_pepper_datasets = DenoisingDataset(salt_pepper_img_array, original_img_array)

# Split data
gaussian_train_datasets, gaussian_test_datasets = random_split(gaussian_datasets, [TRAIN_RATIO, 1 - TRAIN_RATIO])
salt_pepper_train_datasets, salt_pepper_test_datasets = random_split(salt_pepper_datasets, [TRAIN_RATIO, 1 - TRAIN_RATIO])

# Create dataloader
gaussian_dataloader = DataLoader(gaussian_train_datasets, shuffle=True, batch_size=BATCH_SIZE)
salt_pepper_dataloader = DataLoader(salt_pepper_train_datasets, shuffle=True, batch_size=BATCH_SIZE) 






#Lade 1 Bild aus jedem Datensatz
noisy_gaussian, clean_gaussian = next(iter(gaussian_dataloader))
noisy_salt_pepper, clean_salt_pepper = next(iter(salt_pepper_dataloader))
    
    
# Erstelle das Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
print(np.shape(noisy_gaussian))
# Zeige das Bild mit Gaussian Noise
axes[0, 0].imshow(noisy_gaussian.squeeze())
axes[0, 0].set_title('noisy_gaussian')
axes[0, 0].axis('off')
    
axes[0, 1].imshow(clean_gaussian.squeeze())
axes[0, 1].set_title('clean_gaussian')
axes[0, 1].axis('off')
    
# Zeige das Bild mit Salt and Pepper Noise
axes[1, 0].imshow(noisy_salt_pepper.squeeze())
axes[1, 0].set_title('noisy_salt_pepper')
axes[1, 0].axis('off')
    
axes[1, 1].imshow(clean_salt_pepper.squeeze())
axes[1, 1].set_title('clean_salt_pepper')
axes[1, 1].axis('off')
    
plt.tight_layout()
plt.show()