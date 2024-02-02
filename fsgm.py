import os
import random
import csv
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt


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
        self.model.eval()
        image_tensor.requires_grad = True
        outputs = self.model(image_tensor)
        return outputs

    def classify_perturbed_image(self, perturbed_image_tensor):
        """
        Classifies the PERTURBED image based on the provided image tensor
        Returns a (n,1000) tensor with probabilities for each class
        Here, n is the number of images to classify
        """
        self.model.eval()
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
        # Clone the image tensor and set requires_grad to True for the clone
        grad_tensor = image_tensor.clone().detach()
        grad_tensor.requires_grad_(True)

        # Forward pass using grad_tensor
        outputs = self.model(grad_tensor)

        # Process labels and compute loss as before
        labels_tensor = torch.zeros_like(outputs, requires_grad=False, dtype=torch.float32)
        if isinstance(labels, str):
            labels_tensor[0, int(labels) - 1] = 1
        else:
            for i in range(len(labels)):
                labels_tensor[i, int(labels[i]) - 1] = 1

        loss = F.cross_entropy(outputs, labels_tensor)

        # Zero out gradients and backpropagate
        self.model.zero_grad()
        loss.backward()

        # Get the gradient from grad_tensor
        data_grad = grad_tensor.grad.data
        return data_grad

    def fgsm_attack(self, image_tensor, epsilon, data_grad):
        
        """
        Returns a perturbed image based on the provided original image(s),
        epsilon (strength of attack) and gradient of the image(s)
        """
        sign_data_grad = data_grad.sign()
        perturbed_image = self.denormalize(image_tensor) + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return self.normalize(perturbed_image)
    
    def denormalize(self, image_tensor):
        # Denormalize the image tensor
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        
        img = image_tensor.mul(std).add(mean)    # Denormalize
        img = img.clamp(0, 1)                    # Clamp the values to be between 0 and 1
        return img
    
    def normalize(self, image_tensor):
        preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return preprocess(image_tensor)
    
    def adam_attack(self, image_path, epsilon, T, eta=1e-8):
        alpha = epsilon / T
        y = self.get_label(image_path)
        image_t0 = self.denormalize(self.get_img_tensor(image_path))
        print(image_t0)
        image_t = image_t0.clone()
        # image_t = image_t.detach()
        # image_t.requires_grad = True
        g0 = 0 # Redundant
        m0 = 0 # Redundant
        v0 = 0 # Redundant
        dt = 0
        beta1 = 0.9
        beta2 = 0.999

        # Step 3 - Iterate over time
        for t in range(1, T):
            # Step 4 - Find gradient value of step t
            if t == 1:
                gt = self.get_grad(image_t0, self.classify_image(image_t0), y)
            else:
                gt = self.get_grad(image_t, self.classify_perturbed_image(image_t), y)

            # Step 5 & 6- Update biased first and second moment estimate
            if t == 1:
                mt = (1-beta1) * gt
                vt = (1-beta2) * torch.square(gt)
            else: 
                mt = beta1 * mt + (1-beta1) * gt
                vt = beta2 * vt + (1-beta2) * torch.square(gt)

            # Step 7 & 8 - Compute bias-corrected first and second raw moment estimate
            
            mt_hat = mt / (1- beta1 ** t)
            vt_hat = vt / (1- beta2 ** t)

            # Step 9 - Obtain pertubation direction
            dt = mt_hat / (torch.sqrt(vt_hat) + eta)

            # Step 10 - Normalize and scale pertubation vector
            n = torch.prod(torch.tensor(dt.shape))
            
            dt_hat =  n * dt / torch.norm(dt, p=1)

            # Step 11 - Apply pertubation and clip
            if t == 1:
                image_t = torch.clip(image_t0 + alpha * dt_hat, 0, 1) # Correct clipping bounds
            else:
                image_t = torch.clip(image_t + alpha * dt_hat, 0, 1) # Correct clipping bounds

        return self.normalize(image_t)

    
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
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # Tight whitespace
        plt.tight_layout()

        axes[0].imshow(original_img)
        axes[0].axis('off')
        axes[0].set_title("Original image")
        ##axes[0].set_title(f"Predicted class: {original_pred_class} ({original_conf:.3f}). Actual class: {act_class}")
        
        axes[1].imshow(perturbed_img)
        axes[1].axis('off')
        axes[1].set_title("Attacked image")
        #axes[1].set_title(f"Predicted class: {perturbed_pred_class} ({perturbed_conf:.3f})")
        # save figure
        plt.savefig('comparison.png')
        #plt.show()
