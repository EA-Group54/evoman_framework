import numpy as np
import matplotlib.pyplot as plt
import os

# Create figure
horizontal = (12, 6)
vertical = (6, 8)
fig, ax = plt.subplots(1, 2, figsize=horizontal)

# Number of generations
gens = np.arange(0, 50)

groups = {
    0: [],
    1: []
}

# Path to baseline files
base = f'baseline'
for file in os.listdir(base):
    if 'alternativefitness' in file:
        # Load data
        data = np.loadtxt(f'{base}/{file}')

        # Get mean and maximum
        meanfit = np.mean(data, axis=1)
        maxfit = np.max(data, axis=1)

        # Adding to groups
        if '[1, 2, 3, 4, 5, 6, 7, 8]' in file:
            groups[0].append([meanfit, maxfit])
        else:
            groups[1].append([meanfit, maxfit])

labels = ['Average EA1', 'Maximum EA1']
colors = ['blue', 'red']
for i in range(2):
    # Get mean and standard deviation
    meanfit = np.mean(groups[i], axis=0)
    stdfit = np.std(groups[i], axis=0)

    for j in range(2):
        ax[i].plot(gens, meanfit[j], label=labels[j], color=colors[j])
        ax[i].fill_between(gens, meanfit[j] - stdfit[j], meanfit[j] + stdfit[j], color=colors[j], alpha=0.2)

groups = {
    0: [],
    1: []
}

generalist = f'Generalist Main'
for folder in os.listdir(generalist):
    # Path to folder
    folder = f'{generalist}/{folder}'

    # If not a folder
    if not os.path.isdir(folder):
        continue

    # All enemies
    if 'All' in folder:
        # Looping through all files
        for file in os.listdir(folder):
            if not '-fitness' in file:
                continue

            # Read file
            data = np.loadtxt(f'{folder}/{file}')

            # Get mean and maximum
            meanfit = np.mean(data, axis=1)
            maxfit = np.max(data, axis=1)

            # Adding to group
            groups[0].append([meanfit, maxfit])

    # Other group
    else:
        # Looping through all files
        for file in os.listdir(folder):
            if not '-fitness' in file:
                continue

            # Read file
            data = np.loadtxt(f'{folder}/{file}')

            # Get mean and maximum
            meanfit = np.mean(data, axis=1)
            maxfit = np.max(data, axis=1)

            # Adding to group
            groups[1].append([meanfit, maxfit])


labels = ['Average EA2', 'Maximum EA2']
colors = ['green', 'purple']
titles = ['Enemies = 1, 2, 3, 4, 5, 6, 7, 8', 'Enemies = 6, 7, 8']

for i in range(2):
    # Get mean and standard deviation
    meanfit = np.mean(groups[i], axis=0)
    stdfit = np.std(groups[i], axis=0)

    for j in range(2):
        ax[i].plot(gens, meanfit[j], label=labels[j], color=colors[j])
        ax[i].fill_between(gens, meanfit[j] - stdfit[j], meanfit[j] + stdfit[j], color=colors[j], alpha=0.2)

    ax[i].set_title(titles[i])
    ax[i].set_ylim(-100, 10)
    ax[i].set_xlim(0, 50)
    ax[i].legend(loc='upper left', ncol=2)
    ax[i].grid()
    ax[i].set_xlabel('Generations')
    ax[i].set_ylabel('Fitness')
    # if i != 1:
    #     ax[i].xaxis.set_tick_params(labelbottom=False, bottom=False)
    # else:
    #     ax[i].set_xlabel('Generations')

plt.tight_layout()
plt.savefig('Horizontal plots.png', dpi=300)
plt.show()