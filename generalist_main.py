# imports framework
from evoman.environment import Environment

# imports other libs
import numpy as np
import os
from Classes.PlayerController import PlayerController
from Classes.PopulationGeneralistMain import Population


def main(seed, mutation_factor, enemy_list):
    np.random.seed(seed)  # Original seed: 500 # 136 197 296 399 457 ｜ 555 734 814 897 956
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'solutions'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=enemy_list,
                      multiplemode='yes',
                      playermode="ai",
                      player_controller=PlayerController(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    # Number of generations and population size
    popsize = 100
    gen = 50

    # Calculate length for bias and weights array
    n = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    # Bounds for offset and weights
    bounds = (-1, 1)

    # Create first population
    pop = Population(popsize, bounds, n, env, mutation_factor, enemy_list)

    # Run generations
    for i in range(1, gen):
        pop.update(env, 4)
        pop.score(i + 1)

    # Save file
    pop.savefitness(experiment_name, seed)
    pop.saveweights(experiment_name, seed)


if __name__ == '__main__':
    seeds = [136, 197, 296, 399, 457, 555, 734, 814, 897, 956]
    enemies_list = ([1, 2, 3, 4, 5, 6, 7, 8],
                    [6, 7, 8])

    mutation_factor = 0.5
    for seed in seeds:
        for enemies in enemies_list:
            main(seed, mutation_factor, enemies)
