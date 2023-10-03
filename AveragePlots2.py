import numpy as np
import matplotlib.pyplot as plt
import os

# Enemies and path
enemies = [1, 2, 7]
path = 'solutions'

# Create figure
size = 6
fig, ax = plt.subplots(3, 2, figsize=(size * 1.25, size))

gens = np.arange(0, 90)

def getresults(files):
    maxs = []
    avgs = []

    # Looping through each file
    for file in files:
        # Reading data
        file = f'{newpath}/{file}'
        data = np.loadtxt(file)

        # Get max and average from each generation
        max = np.max(data, axis=1)
        avg = np.average(data, axis=1)

        # Add to list
        maxs.append(max)
        avgs.append(avg)

    # Get stds in max and avgs
    maxstd = np.std(maxs, axis=0)
    avgstd = np.std(avgs, axis=0)

    # Calculate average in maximum and average
    maxs = np.average(maxs, axis=0)
    avgs = np.average(avgs, axis=0)

    return maxs, maxstd, avgs, avgstd


for i in range(len(enemies)):
    # Path to enemy
    newpath = f'{path}/enemy{enemies[i]}'

    # Getting non mutated results
    files = [file for file in os.listdir(newpath) if 'fitness' in file and 'm' not in file]
    nomut = getresults(files)

    # Get mutated results
    files = [file for file in os.listdir(newpath) if 'fitness' in file and 'm' in file]
    mut = getresults(files)

    names = ['No mutation', 'Varying Mutation']

    # Plot no mutated results
    ax[i, 0].plot(gens, nomut[0], label='Maximum', color='red')
    ax[i, 0].fill_between(gens, nomut[0] - nomut[1], nomut[0] + nomut[1], color='red', alpha=0.2)
    ax[i, 0].plot(gens, nomut[2], label='Average', color='blue')
    ax[i, 0].fill_between(gens, nomut[2] - nomut[3], nomut[2] + nomut[3], color='blue', alpha=0.2)

    # Plot mutated results
    ax[i, 1].plot(gens, mut[0], label='Maximum', color='red')
    ax[i, 1].fill_between(gens, mut[0] - mut[1], mut[0] + mut[1], color='red', alpha=0.2)
    ax[i, 1].plot(gens, mut[2], label='Average', color='blue')
    ax[i, 1].fill_between(gens, mut[2] - mut[3], mut[2] + mut[3], color='blue', alpha=0.2)

    # Looping through each axis
    for j in range(2):
        ax[i, j].grid()
        ax[i, j].set_ylim(0, 100)
        ax[i, j].set_title(f'Enemy {enemies[i]} ({names[j]})')
        ax[i, j].set_ylabel('Fitness')
        ax[i, j].legend()
        if i != 2:
            ax[i, j].xaxis.set_tick_params(labelbottom=False, bottom=False)
        else:
            ax[i, j].set_xlabel('Generations')



fig.tight_layout()
plt.savefig(f'{path}/plot.png', dpi=300)
plt.show()
