import torch
import torchmetrics
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from skimage.util import random_noise
import copy
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def add_gaussian_noise(img):
    return random_noise(img, mode='gaussian', var=0.5, clip=True)
    #return torch.tensor(img_noisy, dtype=torch.float32)

def add_salt_and_pepper_noise(img, prob=0.5):
    return random_noise(img, mode='s&p', amount=prob, clip=True)
    #return torch.tensor(img_noisy, dtype=torch.float32)

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
        return np.squeeze(img), np.squeeze(img_noisy), label

    def calculate_psnr_ssim(self):
        psnr_values = []
        ssim_values = []

        # Iteriere über den gesamten Datensatz
        for idx in range(len(self)):
            original, noisy, _ = self[idx]

            # Berechne PSNR & SSIM mit Torchmetrics
            psnr_value = psnr(original, noisy, data_range=1)
            ssim_value = ssim(original, noisy, data_range=1, channel_axis=2)

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
        return np.squeeze(img), np.squeeze(img_gaussian), label

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
        return np.squeeze(img), np.squeeze(img_gaussian), label