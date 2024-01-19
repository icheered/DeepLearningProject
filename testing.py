import tkinter as tk
from tkinter import filedialog
from tkinter import Canvas
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
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
model = InceptionV3(weights='imagenet')

# Set random seed
random.seed(2)

def classify_image(file_path):
    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    return decode_predictions(predictions, top=1)

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

loss_fnc = tf.keras.losses.CategoricalCrossentropy()

# FGSM attack code (not working yet)
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# Alternative, also not working yet
def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = model(input_image)
    loss = loss_fnc(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


random_image = choose_random_image()
print("Randomly selected image path:", random_image)
print("Classified as: " + str(classify_image(random_image)))
print(f"True class: {get_label(random_image)}")