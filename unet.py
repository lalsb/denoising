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
BATCH_SIZE = 16
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

# Definiere das U-Net Modell
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Modell, Loss und Optimizer definieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for noisy_imgs, clean_imgs in gaussian_dataloader:
        noisy_imgs, clean_imgs = noisy_imgs.permute(0, 3, 1, 2).to(device).float() / 255.0, clean_imgs.permute(0, 3, 1, 2).to(device).float() / 255.0
        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = criterion(outputs, clean_imgs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(gaussian_dataloader):.4f}")

# Visualisierung eines Beispiels
img_noisy = np.transpose(noisy_imgs[0].cpu().detach().numpy(), (1,2,0))
img_noisy = (img_noisy * 255).astype(np.uint8)
img_noisy = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2RGB)

img_out = np.transpose(outputs[0].cpu().detach().numpy(), (1,2,0))
img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
img_out = (img_out * 255).astype(np.uint8)

img_org = np.transpose(clean_imgs[0].cpu().detach().numpy(), (1,2,0))
img_org = (img_org * 255).astype(np.uint8)
img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_noisy)
axes[0].set_title('Noisy Image')
axes[1].imshow(img_out)
axes[1].set_title('Denoised Image')
axes[2].imshow(img_org)
axes[2].set_title('Original Image')
plt.show()



'''
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
'''