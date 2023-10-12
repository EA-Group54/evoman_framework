from evoman.environment import Environment

# imports other libs
import numpy as np
import os

main_seed = 136
#SEEDED
np.random.seed(main_seed)

population_size = 300 # Aimed population

# Load Individual
experiment_name = 'solutions'
seed = 136
mutation_tag = 'm' # 'm' or ''

file_path = f'{experiment_name}/'+'generalist-weights-'+str(seed)+mutation_tag+'.txt'
individual = np.loadtxt(file_path)

#output file
out_path = f'{experiment_name}/'+'mutated_population.txt'


#Mutate
mutation_factor = .7

# Generat muttant populaiton
holder = []
for i in range(population_size):
    # Add or subtract
    sign = np.random.randint(0, 2, len(individual)) * 2 - 1
    # Add mutation
    mutant = individual + ( (np.random.normal(0, 1, len(individual)) * sign) * mutation_factor )
    #Append to list
    holder.append(mutant)
#Create population array
population = np.array(holder)

print(population)



np.savetxt(out_path, population)