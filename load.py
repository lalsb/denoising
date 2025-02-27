import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

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
#original_gaussian = original_gaussian.squeeze().numpy().transpose(1, 2, 0)
#noisy_gaussian = noisy_gaussian.squeeze().numpy().transpose(1, 2, 0)
    
#original_salt_pepper = original_salt_pepper.squeeze().numpy().transpose(1, 2, 0)
#noisy_salt_pepper = noisy_salt_pepper.squeeze().numpy().transpose(1, 2, 0)
    
# Erstelle das Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
# Zeige das Bild mit Gaussian Noise
axes[0, 0].imshow(np.squeeze(original_gaussian))
axes[0, 0].set_title('Original - Gaussian Noise')
axes[0, 0].axis('off')
    
axes[0, 1].imshow(np.squeeze(noisy_gaussian))
axes[0, 1].set_title('Noisy - Gaussian Noise')
axes[0, 1].axis('off')
    
# Zeige das Bild mit Salt and Pepper Noise
axes[1, 0].imshow(np.squeeze(original_salt_pepper))
axes[1, 0].set_title('Original - Salt & Pepper Noise')
axes[1, 0].axis('off')
    
axes[1, 1].imshow(np.squeeze(noisy_salt_pepper))
axes[1, 1].set_title('Noisy - Salt & Pepper Noise')
axes[1, 1].axis('off')
    
plt.tight_layout()
plt.show()