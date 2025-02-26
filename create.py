import torch
import torchvision.transforms as transforms
from torchvision.datasets import Flowers102
from src.datasets import GaussianNoiseDataset, SaltPepperNoiseDataset
import datetime 

# Gobal vars
gaussian_save_path = "./results/gaussian_noisy_dataset.pt"
salt_pepper_save_path = "./results/salt_pepper_noisy_dataset.pt"

print(f"[{datetime.datetime.now()}] Starting ... ")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = Flowers102(root="./data", split="train", download=True, transform=transform)
val_dataset = Flowers102(root="./data", split="val", download=True, transform=transform)

selected_images = []

# Interleaving the datasets
for train_sample, val_sample in zip(train_dataset, val_dataset):
    selected_images.append(train_sample)  # Append from trainDataset
    selected_images.append(val_sample)    # Append from valDataset
print(f"[{datetime.datetime.now()}] Successfully selected {len(selected_images)} out of 8189 images")

gaussian_dataset = GaussianNoiseDataset(selected_images)
salt_pepper_dataset = SaltPepperNoiseDataset(selected_images)

# Save datasets
torch.save(gaussian_dataset, gaussian_save_path)
torch.save(salt_pepper_dataset, salt_pepper_save_path)
print(f"[{datetime.datetime.now()}] Successfully created Gaussian Noise and Salt & Pepper Noise dataset")

# PSNR und SSIM berechnen
psnr_gaussian, ssim_gaussian = gaussian_dataset.calculate_psnr_ssim()
psnr_sp, ssim_sp = salt_pepper_dataset.calculate_psnr_ssim()

print(f"[{datetime.datetime.now()}] Gaussian Noise:\n [{datetime.datetime.now()}] - PSNR: {psnr_gaussian:.2f} dB\n [{datetime.datetime.now()}] - SSIM: {ssim_gaussian:.4f}")
print(f"[{datetime.datetime.now()}] Salt & Pepper Noise:\n [{datetime.datetime.now()}] - PSNR: {psnr_sp:.2f} dB\n [{datetime.datetime.now()}] - SSIM: {ssim_sp:.4f}")

print(f"[{datetime.datetime.now()}] Done.")

