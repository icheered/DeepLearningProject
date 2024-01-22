import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


pretrained_model = "data/lenet_mnist_model.pth"
use_cuda=True
# Set random seed for reproducibility
torch.manual_seed(42)

# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])),
        batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location=device))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# Implement bim_attack
def bim_attack(model, data, target, epsilon, alpha, num_iter, device):
    """
    Perform the BIM attack and return the effective epsilon.

    Args:
    - model: The neural network model.
    - data: Input data (images).
    - target: The target labels for the input data.
    - epsilon: The maximum amount each pixel can be perturbed.
    - alpha: Step size for each iteration.
    - num_iter: Number of iterations for the attack.
    - device: The device (CPU/GPU) to perform calculations on.
    - early_stop: Whether to stop early if misclassification is achieved.

    Returns:
    - perturbed_data: The perturbed input data after BIM attack.
    - effective_epsilon: The effective epsilon value representing the maximum perturbation.
    """
    perturbed_data = data.clone().detach().to(device)
    perturbed_data.requires_grad = True

    for i in range(num_iter):
        # Forward pass
        output = model(perturbed_data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        if init_pred.item() != target.item():
            break

        loss = F.nll_loss(output, target)

        # Zero gradients
        model.zero_grad()

        # Calculate gradients
        loss.backward()

        # Collect datagrad
        data_grad = perturbed_data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(perturbed_data, alpha, data_grad)

        # Clip the perturbed image to make sure it's within the epsilon-ball
        eta = torch.clamp(perturbed_data - data, min=-epsilon, max=epsilon)
        perturbed_data = torch.clamp(data + eta, 0, 1).detach_()
        if i < num_iter - 1:
            perturbed_data.requires_grad = True

    # Calculate the effective epsilon
    effective_epsilon = (perturbed_data - data).abs().max()

    return perturbed_data, effective_epsilon.item()

# restores the tensors to their original scale
def denorm(batch, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    effective_epsilons = []

    # Loop over the first 100 examples in test set
    num_examples = 100
    for i, (data, target) in enumerate(tqdm(test_loader, desc="Processing", total=num_examples)):
        if i >= num_examples:  # Stop after the first 100 examples
            break

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        
        # Restore the data to its original scale
        data_denorm = denorm(data)

        
        if epsilon == 0:
            perturbed_data = data_denorm
            effective_epsilon = 0
        else:
            # Call BIM Attack
            num_iter = 20
            perturbed_data, effective_epsilon = bim_attack(model, data_denorm, target, epsilon, alpha=epsilon/num_iter, num_iter=num_iter, device=device)
            effective_epsilons.append(effective_epsilon)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1

    # Calculate final accuracy for this epsilon
    #final_acc = correct/float(len(test_loader))
    final_acc = correct/100
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, effective_epsilons

accuracies = []
effective_epsilons = []

# Run test for each epsilon
print("Starting BIM attack...")
epsilons = [0, .05, .1, .15, .2, .25, .3]
#epsilons = [0, .1, .2, .3, .4]
#epsilons = [0, .5] # Maximum peturbation allowed is an epsilon of 0.5
for eps in epsilons:
    accuracy, eff_epsilons = test(model, device, test_loader, eps)
    accuracies.append(accuracy)
    effective_epsilons.append(eff_epsilons)


# Create histogram of effective epsilons for each epsilon
plt.figure(figsize=(5,5))
plt.hist(effective_epsilons, bins=20)
plt.title("Histogram of Effective Epsilons")
plt.xlabel("Effective Epsilon")
plt.ylabel("Count")


# Plot the accuracies for each epsilon
plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()