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