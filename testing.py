import tkinter as tk
from tkinter import filedialog
from tkinter import Canvas
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import torch.nn.functional as F
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

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
model = InceptionV3(weights='imagenet', classes=num_classes)

# Set random seed
random.seed(2)

def get_img_array(file_path):
    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def classify_image(file_path):
    img_array = get_img_array(file_path)

    # Make predictions
    predictions = model.predict(img_array)
    return predictions

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

def get_grad(image_arr, classified_image, label, num_classes=1000):
    classified_image = np.reshape(classified_image, len(classified_image[0]))
    true_softmax = np.zeros_like(classified_image)
    true_softmax[int(label) - 1] = 1
    classified_image_tensor = torch.tensor(classified_image, requires_grad=True, dtype=torch.float32)
    true_softmax_tensor = torch.tensor(true_softmax, dtype=torch.float32)
    loss = F.binary_cross_entropy_with_logits(classified_image_tensor, true_softmax_tensor)
    image_arr_tensor = torch.tensor(image_arr, requires_grad=True, dtype=torch.float32)
    print(f"image_arr_tensor: {image_arr_tensor}")
    loss.backward()
    data_grad = image_arr_tensor.grad
    print(f"Data grad in get_grad: {data_grad}")
    return data_grad

# FGSM attack code (not working yet)
def fgsm_attack(image_arr, epsilon, data_grad):
    image_arr_tensor = torch.tensor(image_arr, dtype=torch.float32, requires_grad=True)
    data_grad = data_grad.to(image_arr_tensor.device)
    sign_data_grad = data_grad.sign()
    perturbed_image = image_arr_tensor + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

random_image = choose_random_image()
classified_image = classify_image(random_image)
print("Randomly selected image path:", random_image)
print(f"Classified as: {decode_predictions(classified_image, top=1)}")
print(f"Vectorized output: {np.argmax(classified_image)+1}")
print(f"True class: {get_label(random_image)}")
image_arr = get_img_array(random_image)

print(f"image arr {image_arr}")

grad = get_grad(image_arr, classified_image, get_label(random_image))

print(f"grad: {grad}")

perturbed_image = fgsm_attack(get_img_array(random_image), 0.1, grad)
print(f"Perturbed image classified as: {decode_predictions(classify_image(perturbed_image.detach().numpy()), top=1)}")

