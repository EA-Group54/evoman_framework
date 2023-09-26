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
enemy = 1   #Interger between 1 and 8



nm_hidden_neurons = 100








env = Environment(experiment_name=experiment_name,
                  enemies=[enemy],
                  playermode="ai",
                  player_controller=PlayerController(nm_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)









def simulation(env,x):
    f,p,e,t = env.play(pcont=x)