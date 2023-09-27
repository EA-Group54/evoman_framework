import numpy as np
import matplotlib.pyplot as plt
import os

# Path to fitness files
path = 'solutions/enemy2'

# Mutation files and non-mutated
mut = []
nomut = []

# Looping through all files within the path
for file in os.listdir(path):
    # Get only txt files with the fitness
    if file.startswith('fitness') and file.endswith('.txt'):
        # Is mutated file
        if 'm' in file:
            data = np.loadtxt(f'{path}/{file}')
            mean = np.mean(data, axis=1)
            maxs = np.max(data, axis=1)
            mut.append([mean, maxs])
        else:
            data = np.loadtxt(f'{path}/{file}')
            mean = np.mean(data, axis=1)
            maxs = np.max(data, axis=1)

            nomut.append([mean, maxs])

# To numpy array
mut = np.array(mut)
nomut = np.array(nomut)

labels = ['Average', 'Maximum']
fig, ax = plt.subplots(1, 2)
ax[0].set_title('Mutation')
ax[1].set_title('No Mutation')

for i in range(2):
    # Plot Mutation
    mean = np.mean(mut[:, i, :], axis=0)
    std = np.std(mut[:, i, :], axis=0)
    gens = np.arange(len(mean))
    ax[0].plot(gens, mean, label=labels[i])
    ax[0].fill_between(gens, mean - std, mean + std, color='lightgrey')

    # Plot no mutation
    mean = np.mean(nomut[:, i, :], axis=0)
    gens = np.arange(len(mean))
    std = np.std(nomut[:, i, :], axis=0)

    ax[1].plot(gens, mean, label=labels[i])
    ax[1].fill_between(gens, mean - std, mean + std, color='lightgrey')

for i in range(2):
    # Add legend and grid
    ax[i].legend()
    ax[i].grid()

    # Set labels
    ax[i].set_xlabel('Generation')
    ax[i].set_ylabel('Fitness')

plt.show()