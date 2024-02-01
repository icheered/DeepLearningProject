from image_classifier import ImageClassifier
import torch
import csv
import os
from tqdm import tqdm
import sys
import random
import string
from torchvision.models.inception import InceptionOutputs
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
    f.write("original_filename,poisoned_filename,label,attack_type,epsilon,steps,prediction,success\n")

# If cmd argument "clean" is given, empty the poisoned_data folder
if len(sys.argv) > 1 and sys.argv[1] == "clean":
    print("CLEANING POISONED_DATA FOLDER")
    remove_counter = 0
    for filename in os.listdir(poisoned_data_folder):
        os.remove(os.path.join(poisoned_data_folder, filename))
        remove_counter += 1
    print(f"Removed {remove_counter} files from {poisoned_data_folder}")

def save_image(original_filename, image_tensor, label, attack_type, epsilon, steps, prediction):
    # Make a random filename of 20 random characters and numbers
    poisoned_filename = "".join(random.choice(letters_and_digits) for i in range(20)) + ".png"
    poisoned_image_path = os.path.join(poisoned_data_folder, poisoned_filename)

    # Transform image tensor to image and write to file
    image = classifier.tensor_to_image(image_tensor)
    image.save(poisoned_image_path)
    
    # Write the original filename to the CSV file in the following format
    # original_filename,poisoned_filename,label,attack_type,epsilon
    with open(poisoned_data_csv, "a") as f:
        f.write(f"{original_filename},{poisoned_filename},{label},{attack_type},{epsilon:.3f},{steps},{prediction},{str(prediction) != str(label)}\n")

# Loop through images
number_of_images =  len(image_paths)

for i in tqdm(range(0, number_of_images)):
    cur_image_paths = image_paths[i:i+1]
    cur_labels = labels[i:i+1]

    # Classify the original image
    image_tensors = classifier.get_img_tensors(cur_image_paths)
    outputs = classifier.classify_image(image_tensors)

    # If output is of type torchvision.models.inception.InceptionOutputs, change it to tensor
    if isinstance(outputs, InceptionOutputs):
        outputs = outputs.logits
    
    _, predicted = outputs.max(1)
    prediction = predicted.item() + 1 # For some reason classifyer is off by 1
    
    # If the prediction is not the same as the label, skip this image
    if str(prediction) != str(cur_labels[0]):
        print(f"Original image misclassified: {cur_image_paths[0]} as {prediction} instead of {cur_labels[0]}")
        continue
    

    cur_original_outputs, cur_original_confs = classifier.decode_predictions(outputs, include_conf=True)
    image_grads = classifier.get_grad(image_tensors, outputs, cur_labels)

    # Apply FGSM attack
    epsilons = [0.01, 0.02, 0.03, 0.04, 0.05]
    for eps in epsilons:
        perturbed_image_tensors = classifier.fgsm_attack(image_tensors, eps, image_grads)
        effective_epsilon = (perturbed_image_tensors - image_tensors).abs().max()
        # Save poisoned image
        for j in range(len(cur_image_paths)): 
            # Original image path is the full path, I only want the filename
            outputs = classifier.classify_image(perturbed_image_tensors)
            _, predicted = outputs.max(1)
            prediction = predicted.item() + 1 # For some reason classifyer is off by 1
            save_image(cur_image_paths[j].split("/")[-1], perturbed_image_tensors[j], cur_labels[j], f"FGSM_{eps}", eps, steps=1, prediction=prediction)


    # Apply BIM attack
    max_epsilon = 0.05
    num_iterations = 10
    perturbed_image_tensors, effective_epsilon, steps, prediction = classifier.bim_attack(image_tensors, max_epsilon, image_grads, cur_labels, num_iterations)
    for j in range(len(cur_image_paths)): 
        save_image(cur_image_paths[j].split("/")[-1], perturbed_image_tensors[j], cur_labels[j], "BIM", effective_epsilon, steps, prediction)

    # # Apply PGD attack
    max_epsilon = 0.05
    num_iterations = 10
    perturbed_image_tensors, effective_epsilon, steps, prediction = classifier.pgd_attack(image_tensors, max_epsilon, image_grads, cur_labels, num_iterations)
    for j in range(len(cur_image_paths)): 
        save_image(cur_image_paths[j].split("/")[-1], perturbed_image_tensors[j], cur_labels[j], "PGD", effective_epsilon, steps, prediction)

    # Show comparison
    # perturbed_outputs = classifier.classify_perturbed_image(perturbed_image_tensors)
    # classifier.show_comparison(image_tensors, perturbed_image_tensors, outputs, perturbed_outputs, cur_labels[0])