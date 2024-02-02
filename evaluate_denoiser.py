import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import csv
from image_classifier import ImageClassifier
import torch
from tqdm import tqdm

def preprocess_image(image_path, target_size=(299, 299)):
    # Load the image
    original_image = load_img(image_path, target_size=target_size)
    # Convert the image to a numpy array
    image = img_to_array(original_image)
    # Scale the image
    image = image / 255.0
    # Expand dimensions to fit model input
    image = np.expand_dims(image, axis=0)
    return original_image, image

def postprocess_image(image):
    # Clip values to be in the range [0, 1]
    image = np.clip(image, 0, 1)
    # Convert to 8-bit pixel values
    image = (255 * image).astype(np.uint8)
    return image

# Get classification rates for every attack type from poisoned_data.csv
poisoned_data_csv = "poisoned_data.csv"
df = pd.read_csv(poisoned_data_csv)

attack_stats_before = []

# original_filename,poisoned_filename,label,attack_type,epsilon,steps,prediction,success
attack_types = df.attack_type.unique()
for attack_type in attack_types:
    # Get the rows with the current attack type
    df_attack_type = df[df.attack_type == attack_type]

    # Calculate the classification rate for the current attack type, where prediction == label
    correct_classification = (df_attack_type['prediction'] == df_attack_type['label']).mean()


    attack_stats_before.append({
        "attack_type": attack_type,
        "classification_rate": correct_classification,
        "avg_epsilon": df_attack_type.epsilon.mean(),
    })

attack_stats_df = pd.DataFrame(attack_stats_before)
# Print the data to terminal
print(attack_stats_df)


# Load the denoiser model
model = load_model(os.path.join("denoiser","denoiser_model_epoch01.keras"))
poisoned_images_dir = "poisoned_data"
all_images = [f for f in os.listdir(poisoned_images_dir) if os.path.isfile(os.path.join(poisoned_images_dir, f))]

# Read the CSV into memory
poisoned_data_csv = "poisoned_data.csv"
poisoned_data_df = pd.read_csv(poisoned_data_csv)

output_csv = "denoiser_evaluation.csv"

# Pull every image through the denoiser, then classify the denoised image and record the result
# In output_csv, write the original_filename, poisoned_filename, label, attack_type, epsilon, steps, prediction


classifier = ImageClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(output_csv, "w") as f:
    f.write("original_filename,poisoned_filename,label,attack_type,prediction_after_denoiser\n")

#for image_name in all_images:
for i in tqdm(range(0, len(all_images))):
    image_name = all_images[i]
    # Use the poisoned_data_df to find the original filename
    original_image_path = None
    current_row = None
    for i, row in poisoned_data_df.iterrows():
        if row['poisoned_filename'] == image_name:
            current_row = row
            original_image_path = os.path.join("dataset/images", row['original_filename'])
            break
    
    if original_image_path is None:
        raise ValueError("No matching original image found in poisoned_data.csv")

    # Preprocess the image
    noisy_image_path = os.path.join(poisoned_images_dir, image_name)
    noisy_image, preprocessed_image = preprocess_image(noisy_image_path)

    # Use the model to predict the denoised image
    denoised_image = model.predict(preprocessed_image)

    # Remove batch dimension
    denoised_image = denoised_image.squeeze()

    # Post-process the image
    denoised_image = postprocess_image(np.array(denoised_image))

    denoised_image_tensor = torch.from_numpy(denoised_image).float()

    # Permute the dimensions to get [C, H, W]
    denoised_image_tensor = denoised_image_tensor.permute(2, 0, 1)  # From (H, W, C) to (C, H, W)

    # Add a batch dimension to get [B, C, H, W]
    denoised_image_tensor = denoised_image_tensor.unsqueeze(0)  # From (C, H, W) to (1, C, H, W)

    # Classify the denoised image
    outputs = classifier.classify_image(denoised_image_tensor)
    _, predicted = outputs.max(1)
    prediction = predicted.item() + 1 # For some reason classifyer is off by 1


    # Write the original filename to the CSV file in the following format
    # original_filename,poisoned_filename,label,attack_type,epsilon,steps,prediction
    with open(output_csv, "a") as f:
        f.write(f"{current_row['original_filename']},{image_name},{current_row['label']},{current_row['attack_type']},{prediction}\n")