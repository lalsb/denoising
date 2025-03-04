import torch
from unet import UNet, DenoisingDataset
from torch.utils.data import Dataset, DataLoader
from utils import calculate_metrics_from_dataset, print_metrics

#model = Unet(*args, **kwargs)
#model.load_state_dict(torch.load("Unet_Gaussian"))

model = UNet()
model.load_state_dict(torch.load("Unet_Gaussian.pth"))
print("loaded Unet_Gaussian Successfully!")
model.eval()

gaus_test = torch.load("Gaussian_test.pt", weights_only=False)
sp_test = torch.load("S&P_test.pt", weights_only=False)

def collate_numpy(batch):
    return batch

gaussian_dataloader = DataLoader(gaus_test, shuffle=True, batch_size=16, collate_fn=collate_numpy)
salt_pepper_dataloader = DataLoader(sp_test, shuffle=True, batch_size=16, collate_fn=collate_numpy)

metrics_gaus = calculate_metrics_from_dataset(gaus_test)
print_metrics(metrics_gaus, "Gaussian_Data")