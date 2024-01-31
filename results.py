import matplotlib.pyplot as plt
import csv
import numpy as np

def import_data(file_path):
    org_output = []
    org_conf = []
    per_output = []
    per_conf = []
    labels = []

    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            org_output.append(row[0])
            org_conf.append(row[1])
            per_output.append(row[2])
            per_conf.append(row[3])
            labels.append(row[4])

    return org_output, org_conf, per_output, per_conf, labels

epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1]
org_num_correct = []    # Number of originally correctly classified images
org_mean_conf = []      # Mean confidence of originally correctly classified images
org_std_conf = []       # Std of originally correctly classified images
per_num_correct = []    # Number of correctly classified perturbed images
per_mean_conf = []      # Mean confidence of correctly classified perturbed images
per_std_conf = []       # Std confidence of correctly classified perturbed images
for i in range(len(epsilons)):
    # Import data and convert to floats
    filename = "results_eps" + str(int(epsilons[i]*100)) + ".csv"
    org_output, org_conf, per_output, per_conf, labels = import_data(filename)
    org_output, org_conf, per_output, per_conf, labels = np.array(org_output, dtype='float32'), np.array(org_conf, dtype='float32'), np.array(per_output, dtype='float32'), np.array(per_conf, dtype='float32'), np.array(labels, dtype='float32')

    # Only keep the images that are classified correctly in the first place
    correctly_classified = np.where(abs(org_output - labels) == 0.)[0]
    org_num_correct.append(len(correctly_classified))
    org_conf = org_conf[correctly_classified]
    per_output = per_output[correctly_classified]
    per_conf = per_conf[correctly_classified]
    labels = labels[correctly_classified]

    org_mean_conf.append(np.mean(org_conf))
    org_std_conf.append(np.std(org_conf))

    # Find which images are perturbed into misclassification and which not
    perturbed = np.where(abs(per_output - labels) > 0.)[0]
    not_perturbed = np.where(abs(per_output - labels) == 0.)[0]
    per_num_correct.append(len(not_perturbed))

    per_conf = per_conf[not_perturbed]
    per_mean_conf.append(np.mean(per_conf))
    per_std_conf.append(np.std(per_conf))

# Add org data as eps=0
epsilons = [0] + epsilons
per_num_correct = [org_num_correct[0]] + per_num_correct
per_mean_conf = [org_mean_conf[0]] + per_mean_conf
per_std_conf = [org_std_conf[0]] + per_std_conf

# Convert to success rate

success_rate = [eps / len(org_output) for eps in per_num_correct]

# Plotting epsilon vs. # of correctly classified perturbed images
plt.figure(figsize=(10, 5))
plt.plot(epsilons, success_rate, marker='o')
plt.title('Correct classification (%) under FSGM attack vs. epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Percentage correctly classified images')
# plt.legend()
plt.grid(True)
plt.show()

# Plotting org_mean_conf vs. epsilons with confidence bands
plt.figure(figsize=(10, 5))

plt.plot(epsilons, per_mean_conf, '-o', label='Mean confidence')

# Define upper and lower bounds for the bands
upper_bound = np.array(per_mean_conf) + np.array(per_std_conf)
lower_bound = np.array(per_mean_conf) - np.array(per_std_conf)

# Fill the area between the upper and lower bounds with a light color
plt.fill_between(epsilons, upper_bound, lower_bound, alpha=0.3, label='standard deviation')

plt.title('Mean confidence of correctly classified images under FSGM attack')
plt.xlabel('Epsilon')
plt.ylabel('Mean confidence')
plt.legend()
plt.grid(True)
plt.show()