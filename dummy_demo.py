from evoman.environment import Environment

# imports other libs
import numpy as np
import os
from Classes.PlayerController import PlayerController

experiment_name = 'optimization_dummy'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=PlayerController(50),
                  enemymode="static",
                  level=2,
                  speed="normal",
                  visuals=True)

# Get best results
sol = np.loadtxt(f'{experiment_name}/firstgen.txt')
print(len(sol))
output = env.play(sol)

