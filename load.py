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

# Dataloade adds 1 in Front of Image Tuple?
gaussian_dataloader = DataLoader(loaded_gaussian_dataset, shuffle=True, batch_size=None)
salt_pepper_dataloader = DataLoader(loaded_salt_pepper_dataset, shuffle=True,batch_size=None)    

#Lade 1 Bild aus jedem Datensatz
original_gaussian, noisy_gaussian, label_gaussian = next(iter(gaussian_dataloader))
original_salt_pepper, noisy_salt_pepper, label_salt_pepper = next(iter(salt_pepper_dataloader))
    
# (1,3,128,128)->(128,128,3)
#original_gaussian = np.squeeze(original_gaussian)
#noisy_gaussian = np.squeeze(noisy_gaussian)
    
#original_salt_pepper = np.squeeze(original_salt_pepper)
#noisy_salt_pepper = np.squeeze(noisy_salt_pepper)
    
# Erstelle das Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
print(np.shape(original_gaussian))
# Zeige das Bild mit Gaussian Noise
axes[0, 0].imshow(original_gaussian)
axes[0, 0].set_title('Original - Gaussian Noise')
axes[0, 0].axis('off')
    
axes[0, 1].imshow(noisy_gaussian)
axes[0, 1].set_title('Noisy - Gaussian Noise')
axes[0, 1].axis('off')
    
# Zeige das Bild mit Salt and Pepper Noise
axes[1, 0].imshow(original_salt_pepper)
axes[1, 0].set_title('Original - Salt & Pepper Noise')
axes[1, 0].axis('off')
    
axes[1, 1].imshow(noisy_salt_pepper)
axes[1, 1].set_title('Noisy - Salt & Pepper Noise')
axes[1, 1].axis('off')
    
plt.tight_layout()
plt.show()