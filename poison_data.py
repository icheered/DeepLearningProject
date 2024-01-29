"""
This file is used to construct the poisoned dataset.
"""
import os

import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm

from mnist_model import MnistModel
from attack import fgsm_attack, bim_attack


# Import dataset
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])),
        batch_size=1, shuffle=True)

# Create folders and files
poisoned_data_folder = "poisoned_data"
poisoned_data_csv = "poisoned_data.csv"

os.makedirs(poisoned_data_folder, exist_ok=True)
if not os.path.exists(poisoned_data_csv):
    with open(poisoned_data_csv, "w") as f:
        f.write("")


# Create model
minst_model = MnistModel()
model = minst_model.get_model()
device = minst_model.get_device()

num_items = 3
for i, (data, target) in enumerate(tqdm(dataloader, desc="Processing", total=num_items)):
    if i >= num_items:
        break

    # Send the data and label to the device
    data, target = data.to(device), target.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    # Forward pass
    output = model(data)
    loss = F.nll_loss(output, target)  # Assuming F is torch.nn.functional

    # Backward pass
    model.zero_grad()
    loss.backward()

    # FGSM attack
    epsilon = 0.3
    perturbed_data = fgsm_attack(data, epsilon, data.grad)
    perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

    # Save perturbed image to poisoned_data folder
    # Filename is the original data filename with the prefix "poisoned_"
    filename = "poisoned_" + os.path.basename(dataloader.dataset.data[i])
    torchvision.utils.save_image(perturbed_data_normalized, os.path.join(poisoned_data_folder, filename))
    