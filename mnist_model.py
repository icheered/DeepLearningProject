import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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


class MnistModel:
    def __init__(self):
        pretrained_model = "data/lenet_mnist_model.pth"
        use_cuda=True

        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Define what device we are using
        print("CUDA Available: ",torch.cuda.is_available())
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

        # Initialize the network
        model = Net().to(device)

        # Load the pretrained model
        model.load_state_dict(torch.load(pretrained_model, map_location=device))

        # Set the model in evaluation mode. In this case this is for the Dropout layers
        model.eval()

        self.device = device
        self.model = model
    
    # restores the tensors to their original scale
    def denorm(self,batch, mean=[0.1307], std=[0.3081]):
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
            mean = torch.tensor(mean).to(self.device)
        if isinstance(std, list):
            std = torch.tensor(std).to(self.device)

        return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

    def get_model(self):
        return self.model
    
    def get_device(self):
        return self.device



import torch
import torch.nn.functional as F





def test( model, device, test_loader, epsilon ):
    # Accuracy counter
    correct = 0
    effective_epsilons = []

    # Loop over the first 100 examples in test set
    num_examples = 1000
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
    final_acc = correct/num_examples
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {num_examples} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, effective_epsilons


def main():
    accuracies = []
    effective_epsilons = []

    # Run test for each epsilon
    print("Starting BIM attack...")
    epsilons = [.5]
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

if __name__ == '__main__':
    main()