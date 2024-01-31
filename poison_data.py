from fsgm import ImageClassifier
import torch
import csv
import os
from tqdm import tqdm
import sys
import random
import string
letters_and_digits = string.ascii_letters + string.digits

# Import dataset
classifier = ImageClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))
image_paths, labels = classifier.get_image_paths(include_labels=True)

# Create folders and files
poisoned_data_folder = "poisoned_data"
poisoned_data_csv = "poisoned_data.csv"

os.makedirs(poisoned_data_folder, exist_ok=True)
with open(poisoned_data_csv, "w") as f:
    f.write("original_filename,poisoned_filename,label,attack_type,epsilon\n")

# If cmd argument "clean" is given, empty the poisoned_data folder
if len(sys.argv) > 1 and sys.argv[1] == "clean":
    print("CLEANING POISONED_DATA FOLDER")
    for filename in os.listdir(poisoned_data_folder):
        os.remove(os.path.join(poisoned_data_folder, filename))
        print(f"Removed {filename}")

# Loop through images
batch_size = 1
number_of_images = 10 # For testing, don't do all images
eps = 0.2
for i in tqdm(range(0, number_of_images, batch_size)):
    cur_image_paths = image_paths[i:i+batch_size]
    cur_labels = labels[i:i+batch_size]

    # Classify the original image
    image_tensors = classifier.get_img_tensors(cur_image_paths)
    outputs = classifier.classify_image(image_tensors)
    cur_original_outputs, cur_original_confs = classifier.decode_predictions(outputs, include_conf=True)

    # Apply attack
    image_grads = classifier.get_grad(image_tensors, outputs, cur_labels)
    perturbed_image_tensors = classifier.fgsm_attack(image_tensors, eps, image_grads)

    # # # # Show comparison
    # # # perturbed_outputs = classifier.classify_perturbed_image(perturbed_image_tensors)
    # # # classifier.show_comparison(image_tensors, perturbed_image_tensors, outputs, perturbed_outputs, cur_labels[0])
    # # # exit()

    # Save poisoned image
    for j in range(len(cur_image_paths)): 
        # Original image path is the full path, I only want the filename
        original_filename = cur_image_paths[j].split("/")[-1]

        # Make a random filename of 20 random characters and numbers
        poisoned_filename = "".join(random.choice(letters_and_digits) for i in range(20)) + ".png"
        poisoned_image_path = os.path.join(poisoned_data_folder, poisoned_filename)
        image_tensor = perturbed_image_tensors[j]

        # Transform image tensor to image and write to file
        image = classifier.tensor_to_image(image_tensor)
        image.save(poisoned_image_path)
        
        # Write the original filename to the CSV file in the following format
        # original_filename,poisoned_filename,label,attack_type,epsilon
        with open(poisoned_data_csv, "a") as f:
            f.write(f"{original_filename},{poisoned_filename},{cur_labels[j]},fgsm,{eps}\n")