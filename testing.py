import tkinter as tk
from tkinter import filedialog
from tkinter import Canvas
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import torch.nn.functional as F
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

import os
import csv
import random

import torch

# Load InceptionV3 model pre-trained on ImageNet
num_classes = 1000
# Load pre-trained InceptionV3 model
model = models.inception_v3(pretrained=True)
model.eval()


# Set random seed
random.seed(2)

# Choose a random image from the dataset
def choose_random_image():
    # Get the current directory of the folder containing images
    script_directory = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_directory, "dataset", "images")

    # Get a list of all images in the folder
    all_files = os.listdir(folder_path)
    image_files = [file for file in all_files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Check if there are any images in the folder
    if not image_files:
        print("No images found in the folder.")
    else:
        # Select a random image path
        random_image_path = os.path.join(folder_path, random.choice(image_files))
    return random_image_path

# Get the true label of the image
def get_label(image_path):
    # Get the current directory of the folder containing images
    script_directory = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_directory, "dataset", "dev_dataset.csv")

    true_label = None

    # Extract the "True label" column from dev_dataset.csv
    try:
        with open(script_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            data = list(reader)
            if len(data) > 0:
                # Extract the 7th and 8th columns as arrays
                images = [row[0] for row in data]
                true_labels = [row[6] for row in data]

                # Extract the filename from the image path
                image_name = os.path.splitext(os.path.basename(image_path))[0]

                # Find the index of the correct image
                try:
                    image_index = images.index(image_name)
                    true_label = true_labels[image_index]
                except ValueError:
                    print(f"'{image_name}' not found in the seventh_column.")
            else:
                print("CSV file is empty.")
    except FileNotFoundError:
        print(f"CSV file not found at {script_path}. Please check the file path.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    return true_label


# Get a image tensor from file path
def get_img_tensor(file_path):
    # Load and preprocess the image
    img = Image.open(file_path)
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor

# Classify image tensor using trained inception-v3 model 
def classify_image(image_tensor):
    # image_tensor = image_tensor.detach()
    image_tensor.requires_grad = True
    outputs = model(image_tensor)
    return outputs

def classify_perturbed_image(perturbed_image_tensor):
    perturbed_image_tensor = perturbed_image_tensor.detach()
    perturbed_image_tensor.requires_grad = True
    outputs = model(perturbed_image_tensor)
    return outputs

# Go from 1x1000 class probabilities to expected class
def decode_predictions(classified_image):
    return torch.argmax(torch.softmax(classified_image, dim=1)) + 1

# Get gradient from image tensor given the output and correct label
def get_grad(image_tensor, outputs, label):
    label_tensor = torch.zeros_like(outputs, requires_grad=False, dtype=torch.float32)
    label_tensor[0, int(label) - 1] = 1
    label_tensor.requires_grad_()
    loss = F.cross_entropy(outputs, label_tensor)
    model.zero_grad()
    loss.backward()
    data_grad = image_tensor.grad.data
    return data_grad

# Perform attack
def fgsm_attack(image_tensor, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image_tensor + epsilon * sign_data_grad
    return perturbed_image

random_image = choose_random_image()
image_tensor = get_img_tensor(random_image)
classified_image = classify_image(image_tensor)
label = get_label(random_image)
print(f"Classified image as: {decode_predictions(classified_image)}, correct class: {label}")
print("Randomly selected image path:", random_image)

data_grad = get_grad(image_tensor, classified_image, label)
print(data_grad.shape)
perturbed_image = fgsm_attack(image_tensor, 0.25, data_grad)
classified_perturbed_image = classify_perturbed_image(perturbed_image)
print(f"Classified perturbed image as: {decode_predictions(classified_perturbed_image)}, correct class: {label}")
print(f"Original image tensor {image_tensor}")
print(f"Perturbed image tensor {perturbed_image}")