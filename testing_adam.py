from fsgm import ImageClassifier
import torch

classifier = ImageClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

image_paths, labels = classifier.get_image_paths(include_labels=True)
random_image, label = classifier.choose_random_image(include_label=True)
img_tensor = classifier.get_img_tensor(random_image)
original_outputs = classifier.classify_image(img_tensor)
perturbed_img_tensor = classifier.adam_attack(random_image, 0.1, 10)
perturbed_outputs = classifier.classify_perturbed_image(perturbed_img_tensor)
# print(img_tensor)
# print(torch.min(img_tensor))
# print(perturbed_img_tensor)
# print(label)
# print(classifier.decode_predictions(perturbed_outputs))
classifier.show_comparison(img_tensor, perturbed_img_tensor, original_outputs, perturbed_outputs, label)