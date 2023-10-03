import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import matplotlib

matplotlib.rc('xtick', labelsize=7.5)

# Path to results
path = 'results for box-plot'

# Dictionary with results
results = {}

# Create figure
fig, ax = plt.subplots()

# Looping through files
for i, file in enumerate(listdir(path)):
    if not file.endswith('.txt'):
        continue

    # Get data
    data = np.loadtxt(f'{path}/{file}')

    # Player and enemy energy
    p = data[:, 0]
    e = data[:, 1]

    # Calculate gain
    gain = p - e

    # Get name
    if file[:-4].endswith('m'):
        enemy = f'Enemy {file[-6]}\n (Varying Mutation)'
    else:
        enemy = f'Enemy {file[-5]}\n (No Mutation)'

    results[enemy] = gain

# Not sure why, in my computer they are coming out of order
# Reordering
results = dict(sorted(results.items()))

# Plot boxplots
ax.boxplot(results.values())

# Set x and y labels
ax.set_xticklabels(results.keys())
ax.set_ylabel('Gain')

# Grid and tight layout for figure
plt.tight_layout()
plt.grid()

plt.savefig(f'{path}/boxplot.png', dpi=300)
plt.show()

# statistical test
from scipy import stats

print("Statistical Test Results:")

stat_test_res = [['Enemy 1\n (No Mutation)', 'Enemy 1\n (Varying Mutation)'], ['Enemy 2\n (No Mutation)', 'Enemy 2\n (Varying Mutation)'], ['Enemy 7\n (No Mutation)', 'Enemy 7\n (Varying Mutation)']]

for enemy in stat_test_res:
    print(f'comparing {enemy[0]}\n to {enemy[1]}')
    # # Generate two sets of data for the two algorithms
    algorithm1_data = np.array(results[enemy[0]])  # Replace with your data
    algorithm2_data = np.array(results[enemy[1]])  # Replace with your data

    # Perform a two-sample t-test
    t_stat, p_value = stats.ttest_ind(algorithm1_data, algorithm2_data)

    # Print the results
    print(f"\tT-statistic:", t_stat)
    print(f"\tP-value:", p_value)

    # Determine the significance based on the p-value
    alpha = 0.05  # You can adjust the significance level as needed
    if p_value < alpha:
        print(f"\tThe difference is statistically significant (reject the null hypothesis)")
    else:
        print(f"\tThe difference is not statistically significant (fail to reject the null hypothesis)")
    print()
