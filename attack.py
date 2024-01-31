import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def bim_attack(model, data, target, epsilon, alpha, num_iter, device):
    loss = nn.CrossEntropyLoss()

    perturbed_data = data.clone().detach().to(device)
    perturbed_data.requires_grad = True

    for i in range(num_iter):
        # Check if noise is already enough to make the model fail
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
        output_normalized = model(perturbed_data_normalized)
        init_pred = output_normalized.max(1, keepdim=True)[1]  # get the index of the max log-probability

        if init_pred.item() != target.item():
            break
        
        # Forward pass
        output = model(perturbed_data)
        loss = loss(output,target)

        # Zero gradients
        model.zero_grad()
        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()

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
            perturbed_data.grad = None

    # Calculate the effective epsilon
    effective_epsilon = (perturbed_data - data).abs().max()

    return perturbed_data, effective_epsilon.item()

def pgd_attack(model, data, target, epsilon, alpha, num_iter, device):
    perturbed_data = data.to(device)
    perturbed_data.requires_grad=True
    target = target.to(device)

    # initialize cross entropy loss class 
    loss = nn.CrossEntropyLoss()

    for i in range(num_iter):
        # Check if noise is already enough to make the model fail
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
        output_normalized = model(perturbed_data_normalized)
        init_pred = output_normalized.max(1, keepdim=True)[1]  # get the index of the max log-probability

        if init_pred.item() != target.item():
            break

        # forward pass
        output = model(perturbed_data)
        # determine loss using predefined cross entropy class
        loss = loss(output,target).to(device)

        # set gradients to zero
        model.zero_grad()
        
        # calculate the gradients
        loss.backward()

        # pgd attack mechanism
        perturbed_data = perturbed_data + alpha*data.grad.sign()
        # clamp perturbed image to epsilon ball
        eta = torch.clamp(perturbed_data - data, min=-epsilon, max=epsilon)
        perturbed_data = torch.clamp(data + eta, min=0, max=1).detach_()

    # determine the effective epsilon
    effective_epsilon = (perturbed_data - data).abs().max()

    return perturbed_data, effective_epsilon.item()