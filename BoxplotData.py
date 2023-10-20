from evoman.environment import Environment
import numpy as np
from Classes.PlayerController import PlayerController
import os


def eval(env: Environment, weights, enemies):
    env.multiplemode = 'no'
    gains = []
    for enemy in enemies:
        env.enemies = [enemy]
        f, p, e, t = env.play(pcont=weights)
        gains.append(p - e)

    return np.mean(gains)


def getResults(p, results):
    for file in os.listdir(p):
        if 'weights' not in file:
            continue

        print(file)
        path = f'{p}/{file}'
        weights = np.loadtxt(path)
        # Which group
        if '[6, 7, 8]' not in file:
            enemies = [1, 2, 3, 4, 5, 6, 7, 8]
            group = 1

        else:
            enemies = [6, 7, 8]
            group = 2

        res = np.mean([eval(env, weights, enemies) for _ in range(10)])
        results[group].append(res)


save = 'boxplot results2'

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

env = Environment(experiment_name=save,
                  enemies=[1],
                  multiplemode='yes',
                  playermode="ai",
                  player_controller=PlayerController(10),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)
base = f'solutions/baseline'

results = {
    1: [],
    2: []
}

# Get results from base
getResults(base, results)
for i in range(2):
    res = results[i + 1]
    np.savetxt(f'{save}/baseline-group{i}', res)

results = {
    1: [],
    2: []
}
new = 'solutions/Generalist Main'
names = ['6-7-8', 'All enemies']
for name in names:
    path = f'{new}/{name}'
    getResults(path, results)

for i in range(2):
    res = results[i + 1]
    np.savetxt(f'{save}/main-group{i}', res)
