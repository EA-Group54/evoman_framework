import numpy as np


def fitness(env, indiv):
    f, p, e, t = env.play(pcont=indiv)
    return f


def mutate(indiv):
    # Add or subtract
    sign = np.random.randint(0, 2, len(indiv)) * 2 - 1

    # Add mutation
    indiv += np.random.normal(0, 1, len(indiv)) * sign

    return indiv


class Population():
    def __init__(self, size, bounds, n, env):
        self.fitness = None
        self.pop = np.random.uniform(bounds[0], bounds[1], (size, n))
        self.savedfitness = []

        # Evaluate fitness
        self.eval(env)

    def eval(self, env):
        self.fitness = list(map(lambda x: fitness(env, x), self.pop))
        self.savedfitness.append(self.fitness)

    def offspring(self, parents):
        """
        Whole Arithmetic Recombination crossover and gaussian mutation
        """

        # Get parents
        p1 = self.pop[parents[0]].flatten()
        p2 = self.pop[parents[1]].flatten()

        # Alpha
        alpha = np.random.uniform(0, 1, len(p1))

        # Create children
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = alpha * p2 + (1 - alpha) * p1

        # Mutation on children
        c1 = mutate(c1)
        c2 = mutate(c2)

        return c1, c2

    def tournament(self, k=10):
        """
        Get best individual based on a tournament
        """
        # Selecting individual
        best = np.random.randint(0, len(self.pop))
        score = self.fitness[best]

        # Going through tournament
        for _ in range(k - 1):
            # Selecting new individual
            new = np.random.randint(0, len(self.pop))

            # Comparing score
            if self.fitness[new] > score:
                best = new
                score = self.fitness[new]

        return best

    def replace(self, indiv, k=10):
        """
        Replaces individuals in population with new one based on tournament
        """
        # Randomly select one individual
        worst = np.random.randint(0, len(self.pop))
        score = self.fitness[worst]

        # Going through tournament
        for _ in range(k - 1):
            # Selecting new individual
            new = np.random.randint(0, len(self.pop))

            # Comparing score
            if self.fitness[new] < score:
                worst = new
                score = self.fitness[new]

        # Replacing with new individual
        self.pop[worst] = indiv

        # Set score high to make it not be replaced by following offspring in same generation
        # self.fitness[worst] = np.inf

    def score(self, gen):
        """
        Get maximum, average and standard deviation from fitness belonging to current population
        """

        maxfit = np.max(self.fitness)
        avgfit = np.mean(self.fitness)
        stdfit = np.std(self.fitness)

        print(f'Gen {gen}: {maxfit}, {avgfit}, {stdfit}')

        # return maxfit, avgfit, stdfit

    def update(self, env, n_child=4):
        """
        Update population with new population
        """

        # Evaluate all individuals in current population
        self.eval(env)

        # Number of children has to be even
        if n_child % 2 != 0:
            n_child += 1

        # Making offspring
        for _ in range(int(n_child / 2)):
            # Getting parents
            p1 = self.tournament()
            p2 = p1
            while p1 == p2:
                p2 = self.tournament()

            # Getting children
            children = self.offspring((p1, p2))

            for child in children:
                self.replace(child)


    def savefitness(self, path):
        # Convert to numpy array
        savedfitness = np.array(self.savedfitness)

        # Save as txt
        np.savetxt(path, savedfitness)

    def saveweights(self, path):
        # Get best fitness
        best = np.where(self.fitness == np.max(self.fitness))

        np.savetxt(path, self.pop[best].flatten())
