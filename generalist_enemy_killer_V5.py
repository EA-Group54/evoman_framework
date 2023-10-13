#Evolution ALgorithm for the EvoMan FrameWork
#Iñigo Auza de la Mora
#Made with the help and guide if the documenation and example code provided by Dr. Karine Miras, karine.smiras@gmail.com

# imports framework
import sys, os

from evoman.environment import Environment
import numpy as np
import random
import statistics
import math


# Changing controller
#from demo_controller import player_controller
from Classes.PlayerController import PlayerController as player_controller

experiment_name = 'generalist_enemy_killer_V5'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


run_mode = 'train' # train or test

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"



#VALUES

#Will be multiply to calculate the number of weights connecting the values
nm_hidden_neurons = 10      #n_hidden_neurons --> nm_hidden_neurons #NOTE@ going to 30 neurons resulted in fitness of 48.018 (way worse)
nm_population = 300     #120 gen: 200 --> 94.9, 500 --> 94.91 600 --> 94.8 #npop --> nm_population
nm_generations = 1000    #120 #gens --> nm_generations
curr_gen = 0
seed_bias = 3   #3  #Any number. Will be used to modify seed
seed_bias2 = 1   #1          #seed_bias=5, seed_bias2=6, seed_bias3=7, 75
seed_bias3 = 4  #4
mutation_factor = .3 #0.4  #MUST BE between 0 and 1
mutation_stepsize = .05
#Limiting values
x_max = 1      #limits
x_min = -1

#initialization values
last_best = 0

#Limiting values
def limits(x):

    if x>x_max:
        return x_max
    elif x<x_min:
        return x_min
    else:
        return x

#Initialize Environment
#Call Environment from evoman.environment, initialize with single enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[1,2,3,4,5,6,7,8],
                  multiplemode='yes',
                  playermode="ai",
                  player_controller=player_controller(nm_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)


# number of weights for multilayer with nm_hidden_neurons
nm_weights = (env.get_num_sensors()+1)*nm_hidden_neurons + (nm_hidden_neurons+1)*5    #n_vars --> nm_weights
print("Number of weights per single layer neural network: ", nm_weights)

#Runs single game (simulates). Apartently env.play() returns four values, we only need the first
def simulate(env,individual):   #Replace simulation with simulate*******
    #print("simulate")   #Errase**** debugging***
    f,p,e,t = env.play(pcont=individual)
    return f

# runs simulation
def simulation(env,indiv):
    _f=[]
    _p=[]
    _e=[]
    _t=[]

    adj_add_e=[]

    k=0
    enemies = [1,2,3,4,5,6,7,8]
    for enemy in enemies:
        new_env = env
        new_env.enemies=[enemy]
        new_env.multiplemode='no'
        f, p, e, t = new_env.play(pcont=indiv)
        #[(x-100)^{2}]/{100} at 0 --> 100, at 50 --> 25, at 100 --> 0   #Closer to e=0, the more points it gets, exponentially
        adjusted_additive_e = ( ((e-100)**2) /100)
        adj_add_e.append(adjusted_additive_e)

        _f.append(f)
        _p.append(p)
        _e.append(e)
        _t.append(t)
        if e <= 0:
            k += 1

    avr_f = statistics.mean(_f)
    avr_p = statistics.mean(_p)
    avr_e = statistics.mean(_e)
    avr_t = statistics.mean(_t)

    return (k*100)+(sum(adj_add_e))+(avr_p/10)

#Functtion evaluate From DEMO code
# evaluation
def evaluate(x):
    #print("evaluate")   #Errase**** debugging***
    return np.array(list(map(lambda y: simulation(env,y), x)))

def evaluate_single(individual):
    #print("evaluate single")   #Errase**** debugging***
    return np.array(list(map(lambda y: simulation(env,y), individual)))

#Look into random.seed() to use a preductable random number****** remember to update the seed***
def compare_n(population, n): #population,
    """
    Compare n contestants from population list
    """
    global seed_bias    #This works

    #Select random individual 
    random.seed(int( str(curr_gen)+str(seed_bias) ) )
    best_contestant = random.randint(0, (population.shape[0]-1) )
    seed_bias = seed_bias+1

    #Ramdom seed from https://www.w3schools.com/python/ref_random_seed.asp
    for nm_individual in range(n-1):
        rand_seed = int(str(curr_gen)+str(nm_individual)+str(seed_bias)) #***could make it longer, add it in string and then turn into interger
        random.seed(rand_seed)
        contestant = random.randint(0, (population.shape[0]-1) )
        seed_bias = seed_bias+1 #We want to make sure that the bias is different if this function is ran multiple times in the same generation 

        if fit_population[contestant]>fit_population[best_contestant]:
            best_contestant = contestant

    return population[best_contestant], mut_population[best_contestant]






def reproduction_n(population, mut_population, nm_parents, nm_participants):
    #placeholdder array for parents
    parents = np.zeros((nm_parents,nm_weights))
    mut_parents = np.zeros(nm_parents)
    for individual in range (nm_parents-1):
        #select eaach parent
        parents[individual], mut_parents[individual] = compare_n(population, nm_participants)
    
    #Creating the offspings
    """
    #Array to contain the offsprings
    curr_offspring = np.zeros(0,nm_parents + 1)
    #mix the offsprings
    variables = np.arange(0,nm_vars,1) #creates the array [0][1][2][3]...[nm_vars]
    for var in nm_vars:
    """
    #we are condensing this step with the next
    #offsprings = parents #********** Create an array with the parents and their values [1,2,3],[4,5,6],[7,8,9] -->
    """
    [1,2,3]
    [4,5,6]
    [7,8,9]
    """
    #and then roll the values np.roll https://numpy.org/doc/stable/reference/generated/numpy.roll.html *******
    offsprings = np.rot90(parents,k=1, axes=(0,1)) # offsprings = np.rot90(parents) ******
    #By rotating it 90 ddegrees, we make the matrix vertigal, with correspinding values for the same variable in the same vertex
    """
    [[3 6 9]
    [2 5 8]
    [1 4 7]]
    """
    #This line is for a latter offspring based on average. NOTE: the offsprings_avr matrix is currently fliped vertical
    offsprings_avr=offsprings

    for i in range(nm_parents-1):   #for every parent
        offsprings[i]=np.roll(offsprings[i],(i%nm_parents)) #This will shift the values in a diagonal way # https://numpy.org/doc/stable/reference/generated/numpy.rot90.html *******
        """
        [[9 3 6]
         [5 8 2]
         [1 4 7]]
        """
    offsprings = np.rot90(offsprings, k=1, axes=(1,0)) #This rotates the matrix in the oposite direction, back to place
    # https://numpy.org/doc/stable/reference/generated/numpy.matrix.mean.html ********
    offsprings_avr = offsprings_avr.mean(1) #NOTE: mean(1) specifies how the average is calculated, if left empty (defaults to mean(0)), itt will return a single value.
    offsprings_avr = offsprings_avr.reshape(1,nm_weights)
    mut_average = mut_parents.mean()

    offsprings = np.concatenate((offsprings,offsprings_avr))
    mut_offspring = np.concatenate((mut_parents,np.reshape(mut_average,1) ))

    return offsprings, mut_offspring


def mutate(population, mutation_factor):
    """
    ****#mutation factor must be a value between 0 and 1 ******* lower is less

    This will be multiplied by a value betwen 0 and 1, which is seeded
    """
    global seed_bias2

    mutants = population
    for individual in population:
        for value in range(nm_weights-1):
            rand_seed = int(str(individual)+str(value)+str(seed_bias2))
            random.seed(rand_seed)
            mutation_random = random.random()*mutation_factor #mutation_factor will specify and reduce the factor by which mutation_random affects the values
            #NOTE: both random.random() and mutation_factor should be between 0 nd 1, which means mutation_random will always be between 0 and 1
            #This will decide if the mutation_random will add or divide. NOTE: the value will always be positive (over 0)
            if random.randint(0,1):
                mutation_random=1+mutation_random
            else:
                mutation_random=1-mutation_random
            mutants[individual][value] = population[individual][value]*mutation_random
            """
            #Alternatively
            rand_seed = int(str(individual)+str(value)+str(seed_bias2))
            random.seed(rand_seed)
            mutation_random = ( random.randint(0,(mutation_factor*10)) /10)
            """
            seed_bias2 = seed_bias2+1
        return mutants


def mutate_individual(individual, mutation_factor):
    """
    ****#mutation factor must be a value between 0 and 1 ******* lower is less

    This will be multiplied by a value betwen 0 and 1, which is seeded
    """
    #print("mutate_individual")   #Errase**** debugging***

    global seed_bias2

    mutant = individual
    for value in range(nm_weights-1):        #nm_sensors or range(population.shape[0]-1)# ***+*+*+**CHECK, might need n_vars - 1
        #print("pass g")    #errase****
        rand_seed = int(str(value)+str(seed_bias2))
        #print(rand_seed)    #errase****
        random.seed(rand_seed)
        mutation_random = random.random()*mutation_factor #mutation_factor will specify and reduce the factor by which mutation_random affects the values
        #NOTE: both random.random() and mutation_factor should be between 0 nd 1, which means mutation_random will always be between 0 and 1
        #This will decide if the mutation_random will add or divide. NOTE: the value will always be positive (over 0)
        if random.randint(0,1):
            mutation_random=1+mutation_random
        else:
            mutation_random=1-mutation_random
        mutant[value] = individual[value]*mutation_random
        """
        #Alternatively
        rand_seed = int(str(individual)+str(value)+str(seed_bias2))
        random.seed(rand_seed)
        mutation_random = ( random.randint(0,(mutation_factor*10)) /10)
        """
        seed_bias2 = seed_bias2+1

    return mutant



def select_survivors(population,fit_population, mut_population):
    #print("select_survivors")   #Errase**** debugging***
    global seed_bias3

    best_to_worse = np.flip(np.argsort(fit_population)) #*****have tto use best_to_worse to know tthe order******+*+*+*+*

    survivors = np.zeros((nm_population,nm_weights))
    fit_survivors = np.zeros((nm_population))
    mut_survivors = np.array(mut_population)

    counter=0
    while counter < (nm_population):
        
        for individual in best_to_worse: #best_to_worse #individual in best_to_worse: #****
            if counter>=nm_population:
                #print("BREAK")
                break
            
            if counter < (nm_population*.05): #NOTE@ Changing from 10 to 2 moved from fitness of 66 to 94 #Change v alue here 0.05 <--> 0.5
                survivors[counter]=population[individual]
                fit_survivors[counter]=fit_population[individual]
                counter=counter+1      
                

            elif counter < (nm_population*9.99/10): #and counter >= (nm_population/10):
                mutant=mutate_individual(population[individual],mut_survivors[individual])
                mutant_fitness = evaluate(mutant.reshape(1, nm_weights)) #*+*+*+*+*++********

                if mutant_fitness > fit_population[individual]: 
                    survivors[counter]=mutant
                    fit_survivors[counter]=mutant_fitness
                    #If the mutation seems to work, incerase it
                    mut_survivors[individual] += mutation_stepsize
                else:
                    survivors[counter]=population[individual]
                    fit_survivors[counter]=fit_population[individual]
                    # If the mutation is too great, decrease it
                    mut_factor = mut_survivors[individual] - mutation_stepsize
                    #But if it becomes negative, reset it
                    if mut_factor < 0:
                        mut_survivors[individual] = mutation_factor
                    else:
                        mut_survivors[individual] = mut_factor
                counter=counter+1
            

            else:   # counter < nm_population:  #*****
                # seeding rand generator numpy ******* https://numpy.org/doc/stable/reference/random/generator.html
                np_random_generator = np.random.default_rng(seed=seed_bias3)
                survivors[counter]=np_random_generator.random((1, nm_weights))
                fit_survivors[counter] = evaluate(survivors[counter].reshape(1, nm_weights))
                seed_bias3 = seed_bias3+1
                counter=counter+1


    return survivors, fit_survivors, mut_survivors
    

def cap_population(population,fit_population, mut_population, population_limit):

    best_to_worse = np.flip(np.argsort(fit_population)) #*****have tto use best_to_worse to know tthe order******+*+*+*+*
    survivors = np.zeros((population_limit,nm_weights))
    fit_survivors = np.zeros((population_limit))
    mut_survivors = np.zeros((population_limit))

    counter=0
    while counter < (nm_population):
        
        for individual in best_to_worse: #best_to_worse #individual in best_to_worse: #****
            if counter>=nm_population:
                #print("BREAK")
                break
            survivors[counter]=population[individual]
            fit_survivors[counter]=fit_population[individual]
            mut_survivors[counter]=mut_population[individual]
            counter=counter+1     

    return survivors, fit_survivors, mut_survivors  


#Frome code example *****
# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)




#Now we really start to run the thing

if not os.path.exists(experiment_name+'/evoman_solstate'):

    np_random_generator = np.random.default_rng(seed=seed_bias3)
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
    if not os.path.exists(experiment_name+'/mutated_population.txt'):
        population = np_random_generator.uniform(x_min,x_max,(nm_population, nm_weights))
        print('New Population in use')
    else:
        population = np.loadtxt(experiment_name+'/mutated_population.txt')
        print('Mutated Population in use')
    #***** multiply e erything by max ***NOPE****
    fit_population = evaluate(population)
    print('TYPE!!!!!: ', type(fit_population))
    mut_population = [mutation_factor]*len(fit_population)

    #FROM DEMO CODE****
    best = np.argmax(fit_population)
    mean = np.mean(fit_population)
    std = np.std(fit_population)
    curr_gen = 0        #ini_g --> curr_gen
    solutions = [population, fit_population]
    env.update_solutions(solutions)



#****from DEMO CODE******
else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()

    #Load previous population ***
    population = env.solutions[0]   #********* 

    #Load previous fit ***
    #fit_population = env.solutions[1]
    fit_population = evaluate(population)

    best = np.argmax(fit_population)
    mean = np.mean(fit_population)
    std = np.std(fit_population)

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    curr_gen = int(file_aux.readline())    #ini_g --> init_generation
    file_aux.close()

    #******add hangling of mutation rate******
    mut_population = [mutation_factor]*len(fit_population)




#****from DEMO CODE******
# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(curr_gen)+' '+str(round(fit_population[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(curr_gen)+' '+str(round(fit_population[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()


# evolution
#****from DEMO CODE******
last_sol = fit_population[best]
notimproved = 0

for i in range(curr_gen, nm_generations+1, 1):


    #Cap Population
    population, fit_population, mut_population = cap_population(population,fit_population,mut_population,nm_population)  #!!!!!!  #****cap population*+*+*+*+*+*
    
    #Addd this, every round will add offspring, to call multipole times reproduction_n
    for r in range(5):
        offspring, mut_offspring = reproduction_n(population, mut_population, 4, round(nm_population*.05)) #+*+*+*+** Increase number of competitors 6 #previous values population, 4, 6
        fit_offspring = evaluate(offspring)   # evaluation
        #DEMO CODE CHECK**********
        #Joins pareents and offspring population
        population = np.vstack((population,offspring))

        #Joins fitness list of parents and offsprings
        fit_population = np.append(fit_population,fit_offspring)
        mut_population = np.append(mut_population,mut_offspring)

    #Individual with highest value
    best = np.argmax(fit_population) #best solution in generation    

    best_sol = fit_population[best]


    # selection
    fit_pop_cp = fit_population 
    fit_pop_norm = fit_pop_cp / np.linalg.norm(fit_pop_cp)
    

    # https://stackoverflow.com/questions/19666626/replace-all-elements-of-numpy-array-that-are-greater-than-some-value ******
    fit_pop_norm[fit_pop_norm < 0] = 0.0000001
    
    probs = (fit_pop_norm)/(fit_pop_norm).sum()

    np_random_generator = np.random.default_rng(seed=seed_bias)


    # searching new areas

    if best_sol <= last_sol:
        notimproved += 1
    else:
        last_sol = best_sol
        notimproved = 0

    if notimproved >= 5:

        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\selection and mutation')
        file_aux.close()

        print("Generational improvement is stalling, mutating individuals")

        population, fit_population, mut_population = select_survivors(population,fit_population, mut_population)    #PROBLEM 2: Should this be here¿¿¿¿
        notimproved = 0

    best = np.argmax(fit_population)
    std = np.std(fit_population)
    mean = np.mean(fit_population)

    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_population[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_population[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name+'/best.txt',population[best])

    # saves simulation state
    solutions = [population, fit_population]
    env.update_solutions(solutions)
    env.save_state()


# From example code*****
file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()

env.state_to_log() # checks environment state

