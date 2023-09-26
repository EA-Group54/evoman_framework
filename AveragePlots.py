import numpy as np
import matplotlib.pyplot as plt
import os

# Path to fitness files
path = 'solutions'

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
    ax[0].plot(np.mean(mut[:, i, :], axis=0), label=labels[i])
    ax[1].plot(np.mean(nomut[:, i, :], axis=0), label=labels[i])

for i in range(2):
    ax[i].legend()
    ax[i].grid()

plt.show()