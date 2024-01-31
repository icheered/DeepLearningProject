import os
import random
import csv
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn


class ImageClassifier:
    def __init__(self, random_seed=True):
        self.model = models.inception_v3(pretrained=True)
        self.model.eval()
        with open("imagenet_classes.txt", "r") as f:
            self.categories = [s.strip() for s in f.readlines()]
        

        # Set seed for reproducability
        if ~random_seed:
            random.seed(2)

    def choose_random_image(self, include_label=False):
        """
        Chooses a random image from the dataset folder
        """
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
        if include_label:
            return random_image_path, self.get_label(random_image_path)
        else:
            return random_image_path
        
    def get_image_paths(self, include_labels=False):
        # Get the current directory of the folder containing images
        script_directory = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(script_directory, "dataset", "images")

        # Get a list of all images in the folder
        all_files = os.listdir(folder_path)
        image_files = [file for file in all_files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print("No images found in the folder.")
        else:
            # Select a random image path
            image_paths = [os.path.join(folder_path, image_file) for image_file in image_files]
        if include_labels:
            labels = []
            for image_path in image_paths:
                labels.append(self.get_label(image_path))
            return image_paths, labels
        else:
            return image_paths
        
    def get_label(self, image_path):
        """
        Gets the true label for the image in the provided image path
        """
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

    def get_img_tensor(self, file_path):
        """
        Returns a (1x3x299x299) torch tensor with gradient
        containing the image from the file path
        """
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
    
    def get_img_tensors(self, file_paths):
        """
        Returns a (Nx3x299x299) torch tensor with gradient
        containing the images from the file paths
        """
        # Load and preprocess the images
        preprocess = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        imgs_tensor = torch.empty(len(file_paths), 3, 299, 299)
        iter = 0
        for path in file_paths:
            img = Image.open(path)
            imgs_tensor[iter] = preprocess(img).unsqueeze(0)
            iter += 1
        return imgs_tensor

    def classify_image(self, image_tensor):
        """
        Classifies the image based on the provided image tensor
        Returns a (n,1000) tensor with probabilities for each class
        Here, n is the number of images to classify
        """
        if image_tensor.ndim == 3:
            # Add a batch dimension if the tensor is 3D
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor.requires_grad = True
        outputs = self.model(image_tensor)
        return outputs


    def classify_perturbed_image(self, perturbed_image_tensor):
        """
        Classifies the PERTURBED image based on the provided image tensor
        Returns a (n,1000) tensor with probabilities for each class
        Here, n is the number of images to classify
        """
        perturbed_image_tensor = perturbed_image_tensor.detach()
        perturbed_image_tensor.requires_grad = True
        outputs = self.model(perturbed_image_tensor)
        return outputs

    def decode_predictions(self, outputs, include_conf=False):
        """
        Returns the class index of the predicted class (1-1000)
        """
        softmax_outputs = torch.softmax(outputs, dim=1)
        if include_conf:
            values, indices = torch.max(softmax_outputs, dim=1)

            return indices + 1, values
        else:
            return torch.argmax(softmax_outputs, dim=1) + 1

    def get_grad(self, image_tensor, outputs, labels):
        """
        Returns the gradient data of the provided image tensors using
        (n, 1000) predictions tensor and true labels (string of indices)
        """
        
        if image_tensor.ndim == 3:
            # Add a batch dimension if the tensor is 3D
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor.requires_grad = True

        labels_tensor = torch.zeros_like(outputs, requires_grad=False, dtype=torch.float32)
        if isinstance(labels, str):
            labels_tensor[0, int(labels) - 1] = 1
        else:
            for i in range(len(labels)):
                labels_tensor[i, int(labels[i]) - 1] = 1
        
        labels_tensor.requires_grad_()
        loss = F.cross_entropy(outputs, labels_tensor)

        self.model.zero_grad()
        loss.backward()

        print(f"Image tensor {image_tensor}")
        print(f"Outputs {outputs}")
        print(f"Labels {labels}")

        data_grad = image_tensor.grad.data
        return data_grad

    def fgsm_attack(self, image_tensor, epsilon, data_grad):
        
        """
        Returns a perturbed image based on the provided original image(s),
        epsilon (strength of attack) and gradient of the image(s)
        """
        sign_data_grad = data_grad.sign()
        perturbed_image = image_tensor + epsilon * sign_data_grad
        #perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image
    
    def bim_attack(self, max_epsilon, num_iter, data, labels):
        alpha = max_epsilon / num_iter
        perturbed_data = data.clone().detach()

        for i in range(num_iter):
            # Forward pass
            outputs = self.classify_image(perturbed_data)

            # Collect data gradient
            data_grad = self.get_grad(perturbed_data, outputs, labels)

            # Call FGSM Attack
            perturbed_data = self.fgsm_attack(perturbed_data, alpha, data_grad)

            # Check if noise is already enough to make the model fail
            outputs = self.classify_image(perturbed_data)
            init_pred = outputs.max(1, keepdim=True)[1]
            if init_pred.item() != labels.item():
                break

        effective_epsilon = (perturbed_data - data).abs().max()
        return perturbed_data, effective_epsilon.item()


        
    
    def tensor_to_image(self, image_tensor):
        """
        Returns a PIL image from the provided image tensor
        """
        # Denormalize the image tensor
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        
        img = image_tensor.squeeze(0)  # Remove the batch dimension
        img = img.mul(std).add(mean)    # Denormalize
        img = img.clamp(0, 1)           # Clamp the values to be between 0 and 1
        img = img.permute(1, 2, 0)      # Rearrange the tensor dimensions to match image format

        # Convert to a PIL image and multiply by 255 as PIL expects pixel values between 0-255
        img = Image.fromarray((img.detach().numpy() * 255).astype('uint8'))
        return img


    def show_image(self, image_tensor, outputs, label):
        # Denormalize the image tensor
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        denormalize = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
        img = denormalize(image_tensor.squeeze(0)).permute(1, 2, 0).detach().numpy()

        # Get category and confidence
        output, conf = self.decode_predictions(outputs, include_conf=True)
        pred_class = self.categories[int(output)-1]
        act_class = self.categories[int(label)-1]

        # Display the image
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image classified as {pred_class} with confidence: {conf:.3f}. Actual class: {act_class}")
        plt.show()
    
    def show_comparison(self, original_image, perturbed_image, outputs, perturbed_outputs, label):
        # Denormalize the image tensor
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        denormalize = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
        original_img = denormalize(original_image.squeeze(0)).permute(1, 2, 0).detach().numpy()
        perturbed_img = denormalize(perturbed_image.squeeze(0)).permute(1, 2, 0).detach().numpy()
        
        # Get category and confidences
        original_output, original_conf = self.decode_predictions(outputs, include_conf=True)
        perturbed_output, perturbed_conf = self.decode_predictions(perturbed_outputs, include_conf=True)
        original_pred_class = self.categories[int(original_output)-1]
        perturbed_pred_class = self.categories[int(perturbed_output)-1]
        act_class = self.categories[int(label)-1]

        # Display the image
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(original_img)
        axes[0].axis('off')
        #axes[0].set_title(f"Predicted class: {original_pred_class} ({original_conf:.3f}). Actual class: {act_class}")
        
        axes[1].imshow(perturbed_img)
        axes[1].axis('off')
        #axes[1].set_title(f"Predicted class: {perturbed_pred_class} ({perturbed_conf:.3f})")
        plt.show()
