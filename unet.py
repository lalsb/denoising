import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from skimage.util import random_noise, img_as_float, img_as_ubyte
from config import ORIGINAL_DIR, GAUSSIAN_DIR, SALT_PEPPER_DIR
from utils import load_images_from_folder, calculate_psnr, calculate_ssim, print_metrics

# Global vars
TRAIN_EPOCHS = 100
TEST_EPOCHS = 36
BATCH_SIZE = 16
TRAIN_RATIO = 0.8
EPOCHS = 10

# Dataset (noisy image, original image)
class DenoisingDataset(Dataset):
    def __init__(self, noisy_images, clean_images):
        self.noisy_images = noisy_images
        self.clean_images = clean_images
    
    def __len__(self):
        return len(self.noisy_images)
    
    def __getitem__(self, i):
        return self.noisy_images[i], self.clean_images[i]

def dataPrep():
    # Load image arrays
    original_img_array = load_images_from_folder(ORIGINAL_DIR)
    gaussian_img_array = load_images_from_folder(GAUSSIAN_DIR)
    salt_pepper_img_array = load_images_from_folder(SALT_PEPPER_DIR)

    # Create datasets
    gaussian_datasets = DenoisingDataset(gaussian_img_array, original_img_array)
    salt_pepper_datasets = DenoisingDataset(salt_pepper_img_array, original_img_array)

    return gaussian_datasets, salt_pepper_datasets

def createTestSplits(gaussian_datasets, salt_pepper_datasets):
    # Split data
    gaussian_train_datasets, gaussian_test_datasets = random_split(gaussian_datasets, [TRAIN_RATIO, 1 - TRAIN_RATIO])
    salt_pepper_train_datasets, salt_pepper_test_datasets = random_split(salt_pepper_datasets, [TRAIN_RATIO, 1 - TRAIN_RATIO])

    # Save Test Splits for Evaluation
    torch.save(gaussian_test_datasets, "Gaussian_test.pt")
    torch.save(salt_pepper_test_datasets, "S&P_test.pt")

    # Create dataloader
    gaussian_dataloader = DataLoader(gaussian_train_datasets, shuffle=True, batch_size=BATCH_SIZE)
    salt_pepper_dataloader = DataLoader(salt_pepper_train_datasets, shuffle=True, batch_size=BATCH_SIZE)

    return gaussian_dataloader, salt_pepper_dataloader

# Definiere das U-Net Modell
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder mit Skip-Connections
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Output-Schicht
        self.final = nn.Conv2d(64, 3, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Encoder-Pfade
        x1 = self.encoder1(x)
        x2 = self.pool1(x1)
        x2 = self.encoder2(x2)
        x3 = self.pool2(x2)
        x3 = self.encoder3(x3)
        x4 = self.pool3(x3)

        # Bottleneck
        x4 = self.bottleneck(x4)

        # Decoder mit Skip-Connections
        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder1(x)

        x = self.final(x)
        return self.activation(x)

def train_Models(gaussian_dataloader, salt_pepper_dataloader):
    # Modell, Loss und Optimizer definieren
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Gaussian Unet
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
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(salt_pepper_dataloader):.4f}")

    torch.save(model.state_dict(), "Unet_Gaussian.pth")

    # Training S&P Unet
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for noisy_imgs, clean_imgs in salt_pepper_dataloader:
            noisy_imgs, clean_imgs = noisy_imgs.permute(0, 3, 1, 2).to(device).float() / 255.0, clean_imgs.permute(0, 3, 1, 2).to(device).float() / 255.0
            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(salt_pepper_dataloader):.4f}")

    torch.save(model.state_dict(), "Unet_S&P.pth")

    # For Examples
    return noisy_imgs, outputs, clean_imgs

def visualize(noisy_imgs, outputs, clean_imgs):
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

def main():

    # Train and save Models
    model_path = "Unet_Gaussian.pth"
    if not os.path.exists(model_path):
        
        gaussian_datasets, salt_pepper_datasets = dataPrep()
        gaussian_dataloader, salt_pepper_dataloader = createTestSplits(gaussian_datasets, salt_pepper_datasets)

        train_Models(gaussian_dataloader, salt_pepper_dataloader)

if __name__ == "__main__":
    main()