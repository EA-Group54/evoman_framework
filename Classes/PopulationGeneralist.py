import numpy as np
import math

"""
def fitness(env, indiv):
    f, p, e, t = env.play(pcont=indiv)
    return f, p, e, t
"""


class Population():
    def __init__(self, size, bounds, n, env, mutation_factor):
        self.currentfitness = None
        self.pop = np.random.uniform(bounds[0], bounds[1], (size, n))
        self.savedfitness = []
        self.factor = mutation_factor
        self.factor_epoch = 1
        self.last_best = 0  # Used to check if improvement was made
        self.counter = 0  # Counter used to reset mutation factor on stall
        self.stall = 5  # Number of epochs of no improvement before reseting the mutation factor

        self.phase = 0
        self.phase_threshold = 80

        # Evaluate fitness
        self.eval(env)

    def fitness(self, env, indiv):
        f, p, e, t = env.play(pcont=indiv)
        """
        print('e',e)
        print('p',p)
        print('t',t)
        print('f',f)
        """
        # If enemy is alive, give negative points for every enemy eneergy
        if e>0:
            print("enemy alive: ", e)
            return (-e)
        if p == 100:
            #If player kills all and survives with 100 p, improve for time
            print("miracle: ", e, p, t)
            return 200 + 100 * math.exp(-0.00307011 * t)
        #If enemy is dead, give 100 for achiving this, plus add player points (Because its the averages, player can win and also loose, having p be negative)
        print("enemy dead: ", e, p)
        return 100 + p  #Alternatively, we could use (p-e) where the runs where the agent loses generate negative p. I believe that currently, we are assuming tthat the enmies are dead because we get negative e. However, if we can get neegative e (overkill an enemy) then it is possibe that some are surviving and the value of others is making the aveerage appear negative.

        """
        # Should read if e > 0 return negative eneemy points, if the player points are 100 p==100, then we care about time, else focus on player points
        if e>0:
            print("enemy alive: ", e)
            return (-e)
        if p == 100:
            print("miracle: ", e, p, t)
            return 100 + 100 * math.exp(-0.00307011 * t)
        print("enemy dead: ", e, p)
        return p
        """
          
        """
        if self.phase == 0:
            return (-e)
        elif self.phase == 1: 
            if (100 - e) != 0:
                return -e
            return p
            # return(100-e)+(p*.1)
        else:
            if p >= self.phase_threshold and e == 0:
                return 100 + 100 * math.exp(-0.00307011 * t)  # math.exp( ( (t)/1000) ) #100( (1/1+x) - 1/3001 )
            return p
        """

    def eval(self, env):
        self.currentfitness = list(map(lambda x: self.fitness(env, x), self.pop))
        self.savedfitness.append(self.currentfitness)

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

    def tournament(self, k=10, avoid=None):
        """
        Get best individual based on a tournament
        """

        # Which to avoid
        if avoid is None:
            avoid = []
        fit = np.delete(self.currentfitness, avoid)

        # Select fitness from k individuals
        scores = np.random.choice(fit, k, replace=False)

        # Get best
        best = np.where(self.currentfitness == np.max(scores))[0][0]

        return best

    def replace(self, indiv, p1, p2, k=10):
        """
        Replaces individuals in population with new one based on tournament
        """

        # Select fitness from k individuals
        scores = np.random.choice(self.currentfitness, k, replace=False)

        # Get worst
        worst = np.where(self.currentfitness == np.min(scores))[0][0]

        # Replacing with new individual
        self.pop[worst] = indiv

        # Set fitness to lowest fitness of parent
        pmin = min(self.currentfitness[p1], self.currentfitness[p2])
        self.currentfitness[worst] = pmin

    def score(self, gen):
        """
        Get maximum, average and standard deviation from fitness belonging to current population
        """

        maxfit = np.max(self.currentfitness)
        avgfit = np.mean(self.currentfitness)
        stdfit = np.std(self.currentfitness)

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
        Update population with new individuals
        """

        ## Having this evaluation first caused the first generation to be updated twice
        # Evaluate all individuals in current population
        # self.eval(env)

        # Number of children has to be even
        if n_child % 2 != 0:
            n_child += 1

        # Making offspring
        for _ in range(int(n_child / 2)):
            # Getting parents
            p1 = self.tournament(10)
            p2 = self.tournament(10, [p1])

            # Getting children
            children = self.offspring((p1, p2))

            for child in children:
                self.replace(child, p1, p2)

        # update mutation factor
        self.update_factor()
        # Start evaluation
        # Evaluate all individuals in current population
        self.eval(env)  # The values are appended after the new children have been added

        """
        if self.phase == 0 and np.max(self.currentfitness) == 0:
            self.phase = 1
            print('entering phase 2/3')
        if self.phase == 1 and np.max(self.currentfitness) == 100:
            self.phase = 2
            print('entering phase 3/3')

        #******
        print("self.phase")
        print(self.phase)
        """

    def savefitness(self, path):
        # Convert to numpy array
        savedfitness = np.array(self.savedfitness)

        # Save as txt
        np.savetxt(path, savedfitness)

    def saveweights(self, path):
        # Get best fitness
        best = np.where(self.currentfitness == np.max(self.currentfitness))[0][0]
        np.savetxt(path, self.pop[best].flatten())

    def update_factor(self):
        self.factor_epoch += 1

    def mutate(self, indiv):
        # Add or subtract
        sign = np.random.randint(0, 2, len(indiv)) * 2 - 1

        # Add mutation
        indiv += ((np.random.normal(0, 1, len(indiv)) * sign) * (self.factor / self.factor_epoch))

        return indiv