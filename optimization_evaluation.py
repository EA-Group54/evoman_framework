# imports framework
import sys, os

import numpy as np
import random

from evoman.environment import Environment
from demo_controller import player_controller

from Classes.PlayerController import PlayerController
from Classes.Population import Population



experiment_name = 'evaluation'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

run_mode = 'train' # train or test

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

#Values
nm_runs = 5
enemy = 7   #Interger between 1 and 8

nm_hidden_neurons = 100

env = Environment(experiment_name=experiment_name,
                  enemies=[enemy],
                  playermode="ai",
                  player_controller=PlayerController(nm_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

#nm_weights = (env.get_num_sensors()+1)*nm_hidden_neurons + (nm_hidden_neurons+1)*5   

def simulation(env,individual):
    f,p,e,t = env.play(pcont=individual)
    return f,p,e,t



directory = 'evaluation'


individual = []

results = {}
results_ = []

for file in os.listdir(directory):
    if file == '.DS_Store' or file == 'evoman_logs.txt':
        #ignore
        print(f'file .DS_Store found!')
    else:
        individual = np.loadtxt(f'{directory}/{file}')
        simulation_trials = []
        trials_p = []
        trials_e = []

        for i in range(nm_runs):
            f,p,e,t = simulation(env, individual)
            simulation_trials.append(f)
            trials_p.append(p)
            trials_e.append(e)


        avg_simulation = ( sum(simulation_trials)/len(simulation_trials) )

        avg_p = ( sum(trials_p)/len(trials_p) )
        avg_e = ( sum(trials_e)/len(trials_e) )

        #results[file] = avg_simulation

        # Gain is defined by average player energy and average enemy energy
        results[file] = [avg_p, avg_e]
        results_.append([avg_p, avg_e])

print(results)
print(results_)


np.savetxt(f'results-enemy{enemy}.txt', np.array(results_))

"""
output = open(f'results-enemy{enemy}.txt', 'w')
output.write(results_)
output.close()
"""






