import torch
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Gobal vars
gaussian_save_path = "./results/gaussian_noisy_dataset.pt"
salt_pepper_save_path = "./results/salt_pepper_noisy_dataset.pt"

def numpy_collate(batch):
    return batch  # Return data as is (NumPy arrays)

def denoise(dataset, dataset_name):
    dataloader = DataLoader(dataset, shuffle=True, batch_size=None, collate_fn=numpy_collate)
    
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
            #original = original.numpy().astype(np.float64)
            #noisy = noisy.numpy().astype(np.float64)
            counter = counter + 1
            if (counter % 100 == 0):
                break

            # Filter anwenden
            denoised_median = median_filter(noisy, size=5)
            denoised_gaussian = gaussian_filter(noisy, sigma=1)
            denoised_mean = uniform_filter(noisy, size=5)

            # PSNR & SSIM berechnen
            psnr_values["median"].append(psnr(original,denoised_median, data_range=1))
            ssim_values["median"].append(ssim(original, denoised_median, data_range=1, channel_axis=2))

            psnr_values["gaussian"].append(psnr(original, denoised_gaussian, data_range=1))
            ssim_values["gaussian"].append(ssim(original, denoised_gaussian, data_range=1, channel_axis=2))

            psnr_values["mean"].append(psnr(original, denoised_mean, data_range=1))
            ssim_values["mean"].append(ssim(original, denoised_mean, data_range=1, channel_axis=2))

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
gaussian_dataset = torch.load(gaussian_save_path, weights_only=False)
salt_pepper_dataset = torch.load(salt_pepper_save_path, weights_only=False)

# Evaluation
denoise(gaussian_dataset, "Gaussian Noise Dataset")
denoise(salt_pepper_dataset, "Salt & Pepper Noise Dataset")
