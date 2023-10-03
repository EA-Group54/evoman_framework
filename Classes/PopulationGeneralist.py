import numpy as np
import math



def fitness(env, indiv):
    f, p, e, t = env.play(pcont=indiv)
    return f,p,e,t

class Population():
    def __init__(self, size, bounds, n, env, mutation_factor):
        self.fitness = None
        self.pop = np.random.uniform(bounds[0], bounds[1], (size, n))
        self.savedfitness = []
        self.factor = mutation_factor
        self.factor_epoch = 1
        self.last_best = 0  # Used to check if improvement was made
        self.counter = 0 # Counter used to reset mutation factor on stall
        self.stall = 5 # Number of epochs of no improvement before reseting the mutation factor

        self.phase=0
        self.phase_threshold=80

        # Evaluate fitness
        self.eval(env)

    def fitness(self, env, indiv):
        f, p, e, t = env.play(pcont=indiv)
        if self.phase == 0:
            return (-e)
        elif self.phase == 1:
            if (100-e) != 0:
                return -e
            return p
            #return(100-e)+(p*.1)
        else:
            if p >= self.phase_threshold and e == 0:
                return 100 + 100*math.exp(-0.00307011(t)) #math.exp( ( (t)/1000) ) #100( (1/1+x) - 1/3001 )
            return p

    def eval(self, env):
        self.fitness = list(map(lambda x: self.fitness(env, x), self.pop))
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

        # No longer needed, now the factor is always used, but a value of 0 results in 0 addition
        """
        if self.factor >= 0:    #if self.factor <= 0:
            # Mutation on children
            c1 = self.mutate(c1)
            c2 = self.mutate(c2)
        """
        c1 = self.mutate(c1)
        c2 = self.mutate(c2)

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

    def score(self, gen):
        """
        Get maximum, average and standard deviation from fitness belonging to current population
        """

        maxfit = np.max(self.fitness)
        avgfit = np.mean(self.fitness)
        stdfit = np.std(self.fitness)

        # If best has not improved, increase counter
        if maxfit <= self.last_best:
            self.counter += 1
        
        # Refereence data for checking progress stalling in the next score() call
        self.last_best = maxfit

        print(f'Gen {gen}: {maxfit}, {avgfit}, {stdfit}')
  
        # If counter has matched stall value, reset the factor epoch, reset counter
        if self.counter >= self.stall:
            print("Progress stalling, reseting mutation factor")
            self.factor_epoch = 1
            self.counter = 0

        # return maxfit, avgfit, stdfit

    def update(self, env, n_child=4):
        """
        Update population with new population
        """

        ## Having this evaluation first caused the first generation to be updated twice
        # Evaluate all individuals in current population
        #self.eval(env)

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

        #update mutation factor
        self.update_factor()

        # Evaluate all individuals in current population
        self.eval(env)  # The values are appended after the new children have been added

        if self.phase == 0 and np.max(self.fitness) == 0:
            self.phase = 1
            print('entering phase 2/3')
        if self.phase == 1 and np.max(self.fitness) == 100:
            self.phase = 2
            print('entering phase 3/3')

    def savefitness(self, path):
        # Convert to numpy array
        savedfitness = np.array(self.savedfitness)

        # Save as txt
        np.savetxt(path, savedfitness)

    def saveweights(self, path):
        # Get best fitness
        best = np.where(self.fitness == np.max(self.fitness))[0][0]
        np.savetxt(path, self.pop[best].flatten())

    def update_factor(self):
        self.factor_epoch += 1

    def mutate(self, indiv):
        # Add or subtract
        sign = np.random.randint(0, 2, len(indiv)) * 2 - 1

        # Add mutation
        indiv += ( (np.random.normal(0, 1, len(indiv)) * sign)*(self.factor/self.factor_epoch) )

        return indiv
    
