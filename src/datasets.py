import torch
import torchmetrics
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from skimage.util import random_noise
import copy

def add_gaussian_noise(img):
    img_np = img.numpy()
    img_noisy = random_noise(img_np, mode='gaussian', var=0.5, clip=True)
    return torch.tensor(img_noisy, dtype=torch.float32)

def add_salt_and_pepper_noise(img, prob=0.5):
    img_np = img.numpy()
    img_noisy = random_noise(img_np, mode='s&p', amount=prob, clip=True)
    return torch.tensor(img_noisy, dtype=torch.float32)

class NoisyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.noisy_dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def add_noise(self, img):
        raise NotImplementedError()

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_noisy = self.add_noise(img.clone())
        return img, img_noisy, label

    def calculate_psnr_ssim(self):
        # Initialisiere die Torchmetrics-Metriken
        psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
        ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

        psnr_values = []
        ssim_values = []

        # Iteriere über den gesamten Datensatz
        for idx in range(len(self)):
            original, noisy, _ = self[idx]

            # Falls nötig, bringe die Bilder auf die richtige Form (B, C, H, W)
            if len(original.shape) == 3:  # Falls einzelne Bilder (C, H, W) → (1, C, H, W)
                original = original.unsqueeze(0)
                noisy = noisy.unsqueeze(0)

            # Berechne PSNR & SSIM mit Torchmetrics
            psnr_value = psnr_metric(original, noisy).item()
            ssim_value = ssim_metric(original, noisy).item()

            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)

        # Berechne den Durchschnitt der PSNR- und SSIM-Werte
        mean_psnr = np.mean(psnr_values)
        mean_ssim = np.mean(ssim_values)

        return mean_psnr, mean_ssim

class GaussianNoiseDataset(NoisyDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset_noisy = copy.deepcopy(dataset)
        for idx in range(len(self)):
            img, label = self.dataset_noisy[idx]
            img = add_gaussian_noise(img)
            self.dataset_noisy[idx] = img, label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_gaussian, _ = self.dataset_noisy[idx]
        return img, img_gaussian, label

class SaltPepperNoiseDataset(NoisyDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset_noisy = copy.deepcopy(dataset)
        for idx in range(len(self)):
            img, label = self.dataset_noisy[idx]
            img = add_salt_and_pepper_noise(img)
            self.dataset_noisy[idx] = img, label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_gaussian, _ = self.dataset_noisy[idx]
        return img, img_gaussian, label