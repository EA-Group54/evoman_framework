import numpy as np
import math
import statistics



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
        """
        _f=[]
        enemies = [1,2,3,4,5,6,7,8]
        for enemy in enemies:
            new_env = env
            new_env.enemies=[enemy]
            new_env.multiplemode='no'
            f, p, e, t = new_env.play(pcont=indiv)
            _f.append(f)

        avr_f = statistics.mean(_f)
        return avr_f
        """

        _f=[]
        _p=[]
        _e=[]
        _t=[]
        k=0
        enemies = [1,2,3,4,5,6,7,8]
        for enemy in enemies:
            new_env = env
            new_env.enemies=[enemy]
            new_env.multiplemode='no'
            f, p, e, t = new_env.play(pcont=indiv)
            _f.append(f)
            _p.append(p)
            _e.append(e)
            _t.append(t)
            if e<=0:
                k+=1

        avr_f = statistics.mean(_f)
        avr_p = statistics.mean(_p)
        avr_e = statistics.mean(_e)
        avr_t = statistics.mean(_t)

        if k <= round(len(enemies)*.6):
            return avr_f, (-avr_e)
        if avr_p <= (60):
            return avr_f, (-avr_e) + avr_p
        return  avr_f, (-avr_e) + avr_p + ( 100 * math.exp(-0.00307011 * avr_t) )   #Formula from 100 to 0 in 3000 steps 100*( math.exp(-t/3000) - (t/(math.exp*3000)) )


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

    def replace(self, indivs, k=10):
        """
        Replaces individuals in population with new one based on tournament
        """
        

        # Select fitness from k individuals
        scores = self.currentfitness.copy()


        # Looping through each individual
        for indiv in indivs:

            # Get random fitness scores
            randscores = np.random.choice(scores, k, replace=False)

            # Get worst
            worst = np.where(self.currentfitness == np.min(randscores))[0][0]

            # Replacing with new individual
            self.pop[worst] = indiv

            # Get worst in scores array
            worst = np.where(scores == np.min(randscores))[0][0]

            # Remove from array
            scores = np.delete(scores, worst)

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

        # List with children
        children = []

        # Making offspring
        for _ in range(int(n_child / 2)):
            # Getting parents
            p1 = self.tournament(10)
            p2 = self.tournament(10, [p1])

            # Add children to list
            for child in self.offspring((p1, p2)):
                children.append(child)

        # Replace individuals with offspring
        self.replace(children)

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
