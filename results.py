import matplotlib.pyplot as plt
import csv
import numpy as np

# Import data from combined_results.csv
data = []
with open("combined_results.csv", mode='r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        data.append(row)

# Get unique epsilons
epsilons = set([float(row[0]) for row in data])

# Sort from small to large
epsilons = sorted(epsilons)

# Split data into lists of lists for each epsilon
data_split = []
for eps in epsilons:
    data_split.append([row for row in data if float(row[0]) == eps])


# Data is like this:
original_num_correct = []    # Number of originally correctly classified images
original_conf_mean = []      # Mean confidence of originally correctly classified images
original_conf_std = []       # Std of originally correctly classified images
perturbed_num_correct = []    # Number of correctly classified perturbed images
perturbed_conf_mean = []      # Mean confidence of correctly classified perturbed images
perturbed_conf_std = []       # Std confidence of correctly classified perturbed images
success_rate = []            # Success rate of attack

for i in range(len(data_split)):
    # Only keep the images that are classified correctly in the first place
    correctly_classified = [row for row in data_split[i] if int(row[1]) == int(row[2])] # Label == original_class
    original_num_correct.append(len(correctly_classified))
    
    original_conf_mean.append(np.mean([float(row[3]) for row in correctly_classified]))
    original_conf_std.append(np.std([float(row[3]) for row in correctly_classified]))

    # Find which images are perturbed into misclassification and which not
    attack_success = [row for row in correctly_classified if int(row[6]) == 1]
    
    # Number of correctly classified perturbed images, i.e. attack fail
    perturbed_num_correct.append(len([row for row in correctly_classified if int(row[1]) == int(row[4])])) 
    
    # Find confidence in correctly attacked images
    perturbed_conf_mean.append(np.mean([float(row[5]) for row in attack_success]))
    perturbed_conf_std.append(np.std([float(row[5]) for row in attack_success]))

    success_rate.append(len(attack_success) / len(correctly_classified))



# Plotting epsilon vs. # of correctly classified perturbed images
plt.figure(figsize=(10, 5))
plt.plot(epsilons, perturbed_num_correct, marker='o')
plt.title('Correct classification (%) under FSGM attack vs. epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Percentage correctly classified images')
plt.grid(True)
# Save image to file
plt.savefig('correct_classification_vs_epsilon.png')


# Plotting org_mean_conf vs. epsilons with confidence bands
plt.figure(figsize=(10, 5))
plt.plot(epsilons, perturbed_conf_mean, '-o', label='Mean confidence')

# Define upper and lower bounds for the bands
upper_bound = np.array(perturbed_conf_mean) + np.array(perturbed_conf_std)
lower_bound = np.array(perturbed_conf_mean) - np.array(perturbed_conf_std)

# Fill the area between the upper and lower bounds with a light color
plt.fill_between(epsilons, upper_bound, lower_bound, alpha=0.3, label='standard deviation')

plt.title('Mean confidence of correctly classified images under FSGM attack')
plt.xlabel('Epsilon')
plt.ylabel('Mean confidence')
plt.legend()
plt.grid(True)
# Save image to file
plt.savefig('mean_confidence_vs_epsilon.png')

