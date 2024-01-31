from fsgm import ImageClassifier
import torch
import csv

classifier = ImageClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

image_paths, labels = classifier.get_image_paths(include_labels=True)

batch_size = 25
print(f"Batch size: {batch_size}, len(image_paths): {len(image_paths)}")

epsilons = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
for eps in epsilons:
    print(f"Epsilon: {eps}")
    original_outputs_tot = []
    original_confs_tot = []
    perturbed_outputs_tot = []
    perturbed_confs_tot = []
    number_of_images = 100
    for i in range(0, number_of_images, batch_size):
        cur_image_paths = image_paths[i:i+batch_size]
        cur_labels = labels[i:i+batch_size]
        image_tensors = classifier.get_img_tensors(cur_image_paths)
        outputs = classifier.classify_image(image_tensors)
        cur_original_outputs, cur_original_confs = classifier.decode_predictions(outputs, include_conf=True)
        original_outputs_tot[i:i+batch_size], original_confs_tot[i:i+batch_size] = cur_original_outputs.detach().numpy(), cur_original_confs.detach().numpy()
        image_grads = classifier.get_grad(image_tensors, outputs, cur_labels)
        perturbed_image_tensors = classifier.fgsm_attack(image_tensors, eps, image_grads)
        perturbed_outputs = classifier.classify_perturbed_image(perturbed_image_tensors)
        cur_perturbed_outputs, cur_perturbed_confs = classifier.decode_predictions(perturbed_outputs, include_conf=True)
        perturbed_outputs_tot[i:i+batch_size], perturbed_confs_tot[i:i+batch_size] = cur_perturbed_outputs.detach().numpy(), cur_perturbed_confs.detach().numpy()
        print(i)

    output_file = "results_eps" + str(int(eps*100)) + ".csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Original Output", "Original Confidence", "Perturbed Output", "Perturbed Confidence", "Label"])
        for i in range(len(original_outputs_tot)):
            writer.writerow([
                original_outputs_tot[i],
                original_confs_tot[i],
                perturbed_outputs_tot[i],
                perturbed_confs_tot[i],
                labels[i]
            ])

print("Done!")


### This is for plotting only one image
random_image, label = classifier.choose_random_image(include_label=True)
img_tensor = classifier.get_img_tensor(random_image)
original_outputs = classifier.classify_image(img_tensor)
img_grad = classifier.get_grad(img_tensor, original_outputs, label)
perturbed_img_tensor = classifier.fgsm_attack(img_tensor, 0.25, img_grad)
perturbed_outputs = classifier.classify_perturbed_image(perturbed_img_tensor)
# classifier.show_image(perturbed_img_tensor, outputs, label)
classifier.show_comparison(img_tensor, perturbed_img_tensor, original_outputs, perturbed_outputs, label)