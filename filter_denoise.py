import torch
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import matplotlib.pyplot as plt

# Gobal vars
gaussian_save_path = "./results/gaussian_noisy_dataset.pt"
salt_pepper_save_path = "./results/salt_pepper_noisy_dataset.pt"

# PSNR & SSIM Initialisierung
psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

def denoise(dataset, dataset_name):
    dataloader = DataLoader(dataset, shuffle=True)
    
    psnr_values = {"median": [], "gaussian": [], "mean": []}
    ssim_values = {"median": [], "gaussian": [], "mean": []}

    # Matplotlib Fenster vorbereiten
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(f"Denoising Example - {dataset_name}")

    example_set = False

    dataloader_iter = iter(dataloader)
    counter = 0

    while True:
        try:
            original, noisy, _ = next(dataloader_iter)
            counter = counter + 1
            if (counter % 100 == 0):
                break

            # Von PyTorch-Tensor zu NumPy-Array konvertieren
            noisy = noisy.squeeze().numpy().transpose(1, 2, 0)

            # Filter anwenden
            denoised_median = median_filter(noisy, size=5)
            denoised_gaussian = gaussian_filter(noisy, sigma=1)
            denoised_mean = uniform_filter(noisy, size=5)

            # Konvertiere NumPy-Arrays in PyTorch-Tensoren
            denoised_median_torch = torch.tensor(denoised_median).unsqueeze(0).permute(0, 3, 1, 2)
            denoised_gaussian_torch = torch.tensor(denoised_gaussian, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
            denoised_mean_torch = torch.tensor(denoised_mean, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)

            # PSNR & SSIM berechnen
            psnr_values["median"].append(psnr_metric(denoised_median_torch, original).item())
            ssim_values["median"].append(ssim_metric(denoised_median_torch, original).item())

            psnr_values["gaussian"].append(psnr_metric(denoised_gaussian_torch, original).item())
            ssim_values["gaussian"].append(ssim_metric(denoised_gaussian_torch, original).item())

            psnr_values["mean"].append(psnr_metric(denoised_mean_torch, original).item())
            ssim_values["mean"].append(ssim_metric(denoised_mean_torch, original).item())

            # Beispielbild setzen
            if not example_set:
                example_set = True

                def show_image(ax, img, title):
                    ax.imshow(img)
                    ax.set_title(title)
                    ax.axis("off")

                show_image(axes[0, 0], noisy, "Noisy Image")
                show_image(axes[0, 1], denoised_median, "Median Filter")
                show_image(axes[1, 0], denoised_gaussian, "Gaussian Filter")
                show_image(axes[1, 1], denoised_mean, "Mean Filter")

        except StopIteration:
            break

    # Durchschnittswerte ausgeben
    print(f"\n=== {dataset_name} ===")
    for method in ["median", "gaussian", "mean"]:
        print(f"{method.capitalize()} Filter - PSNR: {np.mean(psnr_values[method]):.2f}, Filter - SSIM: {np.mean(ssim_values[method]):.2f}")
    plt.show()  # Matplotlib Fenster öffnen

# Datensätze laden
gaussian_dataset = torch.load(gaussian_save_path)
salt_pepper_dataset = torch.load(salt_pepper_save_path)

# Evaluation
denoise(gaussian_dataset, "Gaussian Noise Dataset")
denoise(salt_pepper_dataset, "Salt & Pepper Noise Dataset")
