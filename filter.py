import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter

# Gobal vars
gaussian_save_path = "./results/gaussian_noisy_dataset.pt"
salt_pepper_save_path = "./results/salt_pepper_noisy_dataset.pt"

# Laden
loaded_gaussian_dataset = torch.load(gaussian_save_path, weights_only=False)
loaded_salt_pepper_dataset = torch.load(salt_pepper_save_path, weights_only=False)

gaussian_dataloader = DataLoader(loaded_gaussian_dataset, shuffle=True)
salt_pepper_dataloader = DataLoader(loaded_salt_pepper_dataset, shuffle=True)    

#Lade 1 Bild aus jedem Datensatz
original_gaussian, noisy_gaussian, label_gaussian = next(iter(gaussian_dataloader))
original_salt_pepper, noisy_salt_pepper, label_salt_pepper = next(iter(salt_pepper_dataloader))

# Konvertiere Tensoren zu NumPy-Arrays für die Darstellung
original_gaussian = original_gaussian.squeeze().numpy().transpose(1, 2, 0)
noisy_gaussian = noisy_gaussian.squeeze().numpy().transpose(1, 2, 0)
    
original_salt_pepper = original_salt_pepper.squeeze().numpy().transpose(1, 2, 0)
noisy_salt_pepper = noisy_salt_pepper.squeeze().numpy().transpose(1, 2, 0)

denoised_gaussian_gaus = gaussian_filter(noisy_gaussian, sigma=1)
denoised_gaussian_mean = uniform_filter(noisy_gaussian, size=5)
denoised_gaussian_median = median_filter(noisy_gaussian, size=5)

# Erstelle das Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Zeige das Bild mit Gaussian Noise
axes[0, 0].imshow(noisy_gaussian)
axes[0, 0].set_title('Noisy - Gaussian Noise')
axes[0, 0].axis('off')
    
axes[0, 1].imshow(denoised_gaussian_gaus)
axes[0, 1].set_title('Denoised Gaus')
axes[0, 1].axis('off')
    
# Zeige das Bild mit Salt and Pepper Noise
axes[1, 0].imshow(denoised_gaussian_mean)
axes[1, 0].set_title('Denoised Mean')
axes[1, 0].axis('off')
    
axes[1, 1].imshow(denoised_gaussian_median)
axes[1, 1].set_title('Denoised Median')
axes[1, 1].axis('off')
    
plt.tight_layout()
plt.show()

denoised_sp_gaus = gaussian_filter(noisy_salt_pepper, sigma=1)
denoised_sp_mean = uniform_filter(noisy_salt_pepper, size=5)
denoised_sp_median = median_filter(noisy_salt_pepper, size=5)

# Erstelle das Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Zeige das Bild mit Gaussian Noise
axes[0, 0].imshow(noisy_salt_pepper)
axes[0, 0].set_title('Noisy- SP Noise')
axes[0, 0].axis('off')
    
axes[0, 1].imshow(denoised_sp_gaus)
axes[0, 1].set_title('Denoised Gaus')
axes[0, 1].axis('off')
    
# Zeige das Bild mit Salt and Pepper Noise
axes[1, 0].imshow(denoised_sp_mean)
axes[1, 0].set_title('Denoised Mean')
axes[1, 0].axis('off')
    
axes[1, 1].imshow(denoised_sp_median)
axes[1, 1].set_title('Denoised Median')
axes[1, 1].axis('off')
    
plt.tight_layout()
plt.show()