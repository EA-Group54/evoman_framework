# imports framework
from evoman.environment import Environment

# imports other libs
import numpy as np
import os
from Classes.PlayerController import PlayerController
from Classes.PopulationGeneralistBaseline import Population

def main(seed, mutation_factor, enemy_list):
    seed = seed
    mutation_factor = mutation_factor
    np.random.seed(seed) #Original seed: 500 # 136 197 296 399 457 ｜ 555 734 814 897 956
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
    gen = 90

    # Mutation factor (to decreease overtime, reset on stall)
    #mutation_factor = 0.0 # For experiment, we are using .5 and 0
    # Some results. with gen = 90: .2-->93.4, .5-->93.5, 1-->93.3, 2-->93.9, 3-->93.3, 4-->93.5, 5-->93.9, 12-->93.2, 13-->93.9, 25-->93.4, 50-->93.1 Without mutation 91.9

    # Calculate length for bias and weights array
    n = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    # Bounds for offset and weights
    bounds = (-1, 1)

    # Create first population
    pop = Population(popsize, bounds, n, env, mutation_factor)

    # Run generations
    for i in range(1, gen):
        pop.update(env, 20)
        pop.score(i + 1)

    # Save file
    mutation_tag = ''
    if mutation_factor > 0:
        mutation_tag = 'm'
    pop.savefitness(f'{experiment_name}/'+'-fitness-'+str(seed)+mutation_tag+'.txt')
    pop.saveweights(f'{experiment_name}/'+'-weights-'+str(seed)+mutation_tag+'.txt')


if __name__ == '__main__':

    seeds = [136, 197, 296, 399, 457, 555, 734, 814, 897, 956]
    mutation_factors = [0.5]
    enemies = [1,2,7]


    """
    for seed in seeds:
        print(seed)
        for mutation_factor in mutation_factors:
            print(mutation_factor)
            for enemy in enemies:
                print(enemy)
                main(seed,mutation_factor,enemy)
    """

    seed = 136
    mutation_factor = 0.5
    enemy_list=[1,2,3,4,5,6,7,8]
    main(seed,mutation_factor,enemy_list)