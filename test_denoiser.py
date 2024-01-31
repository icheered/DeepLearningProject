from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def preprocess_image(image_path, target_size=(299, 299)):
    # Load the image
    image = load_img(image_path, target_size=target_size)
    # Convert the image to a numpy array
    image = img_to_array(image)
    # Scale the image
    image = image / 255.0
    # Expand dimensions to fit model input
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_image(image):
    # Clip values to be in the range [0, 1]
    image = np.clip(image, 0, 1)
    # Convert to 8-bit pixel values
    image = (255 * image).astype(np.uint8)
    return image

# Load the model
model = load_model("denoiser_model.keras")


# List all files in the poisoned_images_dir
poisoned_images_dir = "poisoned_data"
all_images = [f for f in os.listdir(poisoned_images_dir) if os.path.isfile(os.path.join(poisoned_images_dir, f))]
if not all_images:
    raise ValueError("No images found in the directory")

# Select a random image from the directory
random_image_name = random.choice(all_images)
image_path = os.path.join(poisoned_images_dir, random_image_name)


# Preprocess the image
input_image = preprocess_image(image_path)

# Use the model to predict the denoised image
denoised_image = model.predict(input_image)

# Remove batch dimension
denoised_image = denoised_image.squeeze()

# Post-process the image
denoised_image = postprocess_image(denoised_image)

# Load the original image for display
original_image = load_img(image_path, target_size=(299, 299))

# Display the original and denoised images
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(denoised_image)
plt.title("Denoised Image")
plt.axis("off")

plt.show()