from evoman.environment import Environment

# imports other libs
import numpy as np
import os
from Classes.PlayerController import PlayerController

experiment_name = 'solutions'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
def def_env(enemy):
    env = Environment(experiment_name=experiment_name,
                    enemies=[enemy],
                    multiplemode='no',
                    playermode="ai",
                    player_controller=PlayerController(10),
                    enemymode="static",
                    level=2,
                    speed="normal",
                    visuals=True)
    return env

# Get best results
sol = np.loadtxt(f'{experiment_name}/generalist-weights-136m.txt')
print(len(sol))

results = []
enemies = [1,2,3,4,5,6,7,8]
for enemy in enemies:
    env = def_env(enemy)
    output = env.play(sol)
    results.append(output)
    print(output)

print('results: ')
for list in results:
    print('p: ',list[1], 'e: ',list[2],  't; ', list[3] )


print('results: ')
for list in results:
    print(list[1],list[2], list[3] )