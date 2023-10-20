import numpy as np
import matplotlib.pyplot as plt
import os

# Dict with data
groups = {
    0: {},  # Enemies [1,2,3,4,5,6,7,8]
    1: {},  # Enemies [6,7,8]
}

# Names of EA
names = ['EA1', 'EA2'] * 2

# Getting data by looping through file
for file in os.listdir():
    if 'group' not in file:
        continue
    data = np.loadtxt(file)
    group = int(file.split('group')[1])

    if 'baseline' in file:
        name = names[0]
        groups[group]['baseline'] = data
    else:
        name = names[1]
        groups[group]['main'] = data


# Keys of dict
keys = ['baseline', 'main']

# List for boxplots
bps = [None, None]

# Used colors for boxplots
colors = ['blue', 'green']

# Plotting
fig, ax = plt.subplots()
for i in range(2):
    for j, key in enumerate(keys):
        data = groups[i][key]
        pos = i * 2 + j
        bps[i] = ax.boxplot(data, positions=[pos],
                            patch_artist=True, boxprops={'facecolor': colors[i]})
ax.set_xticklabels(names)

# Enemies and adding legend
enemies = ['[1,2,3,4,5,6,7,8]', '[6,7,8]']
ax.legend([bps[0]["boxes"][0], bps[1]["boxes"][0]], enemies, loc='upper right')

# Setting labels
ax.set_xlabel('Evolutionary Algorithms')
ax.set_ylabel('Average gain')

plt.savefig('Boxplot.png', dpi=300)
plt.show()
