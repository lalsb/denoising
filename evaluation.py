import torch
from unet import UNet, DenoisingDataset
from torch.utils.data import Dataset, DataLoader
from utils import calculate_metrics_from_dataset, print_metrics, calculate_psnr, calculate_ssim
import numpy as np
import cv2
from filter import denoise_and_evaluate_dataset as filter
from filter import denoise_and_evaluate_dataset as fourier

gaus_test = torch.load("Gaussian_test.pt", weights_only=False)
sp_test = torch.load("S&P_test.pt", weights_only=False)

def collate_numpy(batch):
    return batch

gaussian_dataloader = DataLoader(gaus_test, shuffle=True, batch_size=16)
salt_pepper_dataloader = DataLoader(sp_test, shuffle=True, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Evaluate the gaussian test dataset using the metrics 
model = UNet()
model.load_state_dict(torch.load("Unet_Gaussian.pth"))
print("loaded Unet_Gaussian Successfully!")
model.eval()
model.to(device)

metrics = {
    'original': {'psnrs': [], 'ssims': []}
    }
with torch.no_grad():
    for noisy_imgs, original_imgs in gaussian_dataloader:
        noisy_imgs, original_imgs = noisy_imgs.permute(0, 3, 1, 2).to(device).float() / 255.0, original_imgs.permute(0, 3, 1, 2).to(device).float() / 255.0

        denoised_imgs = model(noisy_imgs)

        for i in range(denoised_imgs.cpu().numpy().shape[0]):
            img_out = np.transpose(denoised_imgs[i].cpu().detach().numpy(), (1,2,0))
            img_out = (img_out * 255).astype(np.uint8)

            img_org = np.transpose(original_imgs[i].cpu().detach().numpy(), (1,2,0))
            img_org = (img_org * 255).astype(np.uint8)

            psnr = calculate_psnr(img_org, img_out)
            ssim = calculate_ssim(img_org, img_out)
            metrics['original']['psnrs'].append(psnr)
            metrics['original']['ssims'].append(ssim)

metric_gaus = print_metrics(metrics, "Gaus")

# Evaluate the salt and pepper test dataset using the metrics
model = UNet()
model.load_state_dict(torch.load("Unet_S&P.pth"))
print("loaded Unet_S&P Successfully!")
model.eval()
model.to(device)

metrics = {
    'original': {'psnrs': [], 'ssims': []}
    }
with torch.no_grad():
    for noisy_imgs, original_imgs in salt_pepper_dataloader:
        noisy_imgs, original_imgs = noisy_imgs.permute(0, 3, 1, 2).to(device).float() / 255.0, original_imgs.permute(0, 3, 1, 2).to(device).float() / 255.0

        denoised_imgs = model(noisy_imgs)

        for i in range(denoised_imgs.cpu().numpy().shape[0]):
            img_out = np.transpose(denoised_imgs[i].cpu().detach().numpy(), (1,2,0))
            img_out = (img_out * 255).astype(np.uint8)

            img_org = np.transpose(original_imgs[i].cpu().detach().numpy(), (1,2,0))
            img_org = (img_org * 255).astype(np.uint8)

            psnr = calculate_psnr(img_org, img_out)
            ssim = calculate_ssim(img_org, img_out)
            metrics['original']['psnrs'].append(psnr)
            metrics['original']['ssims'].append(ssim)

metric_gaus = print_metrics(metrics, "S&P")

def evaluate_conventional_methods():
    gaus_filter_metric = filter(gaus_test)
    sp_filter_metric = filter(sp_test)

    gaus_fourier_metric = fourier(gaus_test)
    sp_fourier_metric = fourier(sp_test)

    print_metrics(gaus_filter_metric)
    print_metrics(sp_filter_metric)
    print_metrics(gaus_fourier_metric)
    print_metrics(sp_fourier_metric)

evaluate_conventional_methods()