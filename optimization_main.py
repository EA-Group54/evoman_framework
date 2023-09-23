# imports framework
from evoman.environment import Environment

# imports other libs
import numpy as np
import os
from Classes.PlayerController import PlayerController
from Classes.Population import Population

def main():
    np.random.seed(500)
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'solutions'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 100

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=[2],
                      playermode="ai",
                      player_controller=PlayerController(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    # Number of generations and population size
    popsize = 25
    gen = 90

    # Mutation factor (to decreease overtime, reset on stall)
    mutation_factor = .2 # Some results. with gen = 90: .2-->93.4, .5-->93.5, 1-->93.3, 2-->93.9, 3-->93.3, 4-->93.5, 5-->93.9, 12-->93.2, 13-->93.9, 25-->93.4, 50-->93.1 Without mutation 91.9

    # Calculate length for bias and weights array
    n = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    # Bounds for offset and weights
    bounds = (-1, 1)

    # Create first population
    pop = Population(popsize, bounds, n, env, mutation_factor)

    for i in range(1, gen):
        pop.update(env, 20)
        pop.score(i + 1)

    pop.savefitness(f'{experiment_name}/fitness.txt')


if __name__ == '__main__':
    main()
