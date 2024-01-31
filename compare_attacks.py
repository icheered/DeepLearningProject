

import matplotlib.pyplot as plt
import pandas as pd
import os

# Import poisoned_data.csv
poisoned_data_csv = "poisoned_data.csv"
df = pd.read_csv(poisoned_data_csv)

# original_filename,poisoned_filename,label,attack_type,epsilon,steps,prediction,success
# 8.png,c.png,389,FGSM_0.1,0.100,1,388,True
# 8.png,3.png,389,FGSM_0.2,0.200,1,389,False
# 8.png,l.png,389,FGSM_0.3,0.300,1,389,False
# 8.png,L.png,389,BIM,0.030,0,388,True
# 8.png,G.png,389,PGD,0.300,1,372,True

# Plot the number of correctly classified images for each attack type

attack_stats = []

attack_types = df.attack_type.unique()
for attack_type in attack_types:
    # Get the rows with the current attack type
    df_attack_type = df[df.attack_type == attack_type]
    # Get the rows where the attack was successful
    df_attack_type_success = df_attack_type[df_attack_type.success == True]
    # Get the number of correctly classified images
    num_correct = len(df_attack_type_success)
    # Get the total number of images
    num_total = len(df_attack_type)

    # Average epsilon
    avg_epsilon = df_attack_type.epsilon.mean()

    # Average steps
    avg_steps = df_attack_type.steps.mean()
    
    attack_stats.append({
        "attack_type": attack_type,
        "success_rate": num_correct / num_total,
        "avg_epsilon": avg_epsilon,
        "avg_steps": avg_steps
    })

# Plot the number of correctly classified images for each attack type using plt
#plt.figure()
attack_stats_df = pd.DataFrame(attack_stats)
attack_stats_df.plot.bar(x="attack_type", y="success_rate", rot=0)


# Plot the average epsilon for each attack type using plt
#plt.figure()
attack_stats_df.plot.bar(x="attack_type", y="avg_epsilon", rot=0)
plt.show()
