from random import randint, random
from operator import add
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
import math
from statistics import median, mode
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def individual(length, min, max):
    "Create a member of the population."
    return [ randint(min,max) for x in range(length)]

def population(count, length, min, max):
    return [ individual(length, min, max) for x in range(count) ]

def poly_comp(x, coeffecient): 
    result = coeffecient[0]
    for i in range(1, len(coeffecient)):
        result = result + coeffecient[i] * x**i
    return result

def fitness(individual, target):
    """
        Determine the fitness of an individual. Lower is better. 
        individual: the individual to evaluate
        target: the target number individuals are aiming for
    """
    targetArray = np.array(target); indArray = np.array(individual) #Convert individual and targets into arrays
    sum = np.sum(abs(np.subtract(targetArray,indArray)))
    return sum             #Return an average fitness for all coordinates using individual's genes

def grade(pop, target, bestFitness, best_individual):
    'Find average fitness for a population.'
    fitnessList = []
    for x in range(len(pop)):
        IndFit = fitness(pop[x], target)                             #Calculate fitness of individual in population
        if IndFit < bestFitness or IndFit == 0:                        #Record the best fitness and individual in population
            bestFitness = IndFit; best_individual = pop[x]
        fitnessList.append(IndFit)
    
    #fitnessList = reject_outliers(np.array(fitnessList), 2)
    summed = (reduce(add, fitnessList))/(len(pop) * 1.0)
    list_mode = mode(fitnessList)
    
    return summed, best_individual, bestFitness, list_mode

def evolve(pop, target, populationFitness, type, elitism, c, retain=0.1, random_select=0.05, mutate1=0.3, mutate2=0.3): ##BIG NOTICE: these vals can be changed to change optimizer 
    
    graded = [ (fitness(x, target), x) for x in pop] #Create a 2D list
    #####
    if type == 'roullete': #ROULETTE

        #Invert fitness values
        invertedFit = [ 1/(0.1+i[0]) for i in graded ]

        # number of retained individuals
        retain_length = int(len(graded)*retain)
        parents = []

        # calculate sum of all fitnessees, determine % on wheel
        #store in list
        total = sum(invertedFit)
        fitnessRoul = [ i/total for i in invertedFit ]
        probabilityDist = [sum(fitnessRoul[:i+1]) for i in range(len(fitnessRoul))] # probability distribution

        # Spin
        for i in range(retain_length):
            selector = random()
            for (i,individual) in enumerate(pop):
                if selector <= probabilityDist[i]:
                    parents.append(individual)
                    break
    
    elif type == 'ranked': # RANKED

        #Rank then retain top % individuals
        graded = [ x[1] for x in sorted(graded) ]
        retain_length = int(len(graded)*retain)
        parents = graded[:retain_length]

        # Maintain genetic diversity by adding other individuals
        for individual in graded[retain_length:]:
            if random_select > random():
                parents.append(individual)



    #####
    
    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length #Finds how many spaces are left in population outside of the parent chromosones
    children = []

    while len(children) < desired_length:
        male = randint(0, parents_length-1) #Choose random parents out of parents list
        female = randint(0, parents_length-1) 
        if male != female: #making sure both parents aren't the same
            male = parents[male]
            female = parents[female]

            ######### CROSS OVER #########
            if c == 1:
                ### 1 Point cross-over
                point = randint(1, (len(male)-2))      
                child = male[:int(point)] + female[int(point):] 
                children.append(child)      
            elif c == 2:
                ### 2 Point cross-over
                third_point = int(len(male)/3)
                point1 = randint(1, third_point); point2 = randint(point1,(len(male)-1)) 
                child = male[:point1] + female[point1:point2] + male[point2:]
                children.append(child) 
            elif c == 3:
                ### Uniform cross-over
                child = []
                for x in range(len(male)):
                    selector = randint(0,1)
                    if selector == 0:
                        child.append(male[x])
                    elif selector == 1:
                        child.append(female[x])
                    children.append(child)

    parents.extend(children) #Fills spaces left in population size with children generated via crossover by parents

    # mutate some individuals

    n = 0

    # only use elitism if selection is done through ranking
    if elitism == 1:
        # calculate number of elite members to pass onto the next generation
        n = int(0.05*len(parents))
        # select elite members
        elite = parents[:n]
        parents = parents[n:]

    for individual in parents:

        if fitness(individual, target) > populationFitness:
            if mutate1 > random():
                pos_to_mutate = randint(0, len(individual)-1) #Chooses a random number within individual list
                individual[pos_to_mutate] = randint(min(individual), max(individual))
        elif fitness(individual, target) < populationFitness:
            if mutate2 > random():
                pos_to_mutate = randint(0, len(individual)-1) #Chooses a random number within individual list
                individual[pos_to_mutate] = randint(min(individual), max(individual)) 

    if elitism == 1:
        parents = elite + parents   

    return parents

### MAIN CODE ###

target = [-19, 7, -14, 31, 18, 25]

p_count = 500
i_length = 6
i_min = -40
i_max = 40
generations = 1000

start_value = 1
end_value = 0
iterations = 50


### Fixed Generation End Condition ###
best_individual = []
best_fitness = 5e+30
fitness_history = []
grade_population = []
seeded_fitness_history = []
best_individual_history = []
mode_history = []

type = 'ranked'
elitism = 0
Av_final = []
M_final = []
b_final = []
best_individual = []
best_fitness = 5e+30

bestInd_allruns = []

pop_orig = population(p_count, i_length , i_min , i_max )

graded = [(fitness(x, target), x) for x in pop_orig] #Create a 2D list
graded = [x[1] for x in sorted(graded)]
bestInd_allruns = [fitness(graded[0], target),graded[0]]
converged_list = []
converged_averaged = []

for x in range(iterations):
    varying_parameter = start_value - start_value*x/iterations
    mutation_count = varying_parameter
    print(varying_parameter)
    pop_R = pop_orig
    fitness_history_R = []
    mode_history_R = []
    best_fitness_R = []
    population_fitness, best_individual, best_fitness, fitness_mode = grade( pop_orig , target, best_fitness, best_individual)
    fitness_history_R.append(population_fitness)
    mode_history_R.append(fitness_mode)
    best_fitness_R.append(best_fitness)

    accepted = 0
    stop = 1
    for y in range(generations):
        pop_R = evolve( pop_R , target, fitness_history_R[y-1], type, elitism, 1, mutate1=mutation_count, mutate2= mutation_count)
        population_fitness, best_individual, best_fitness, fitness_mode = grade( pop_R, target, best_fitness, best_individual)
        if population_fitness <=10 and stop != 0:
            converged_list.append([y, varying_parameter])
            stop = 0

        fitness_history_R.append(population_fitness); mode_history_R.append(fitness_mode); best_fitness_R.append(best_fitness)
    # print(converged_list)
    if best_fitness <= bestInd_allruns[0]:
        bestInd_allruns = [best_fitness, best_individual]

    if x == 0:
        b_final = best_fitness_R
        Av_final = fitness_history_R
        M_final = mode_history_R
    
    else:
        for y in range(len(b_final)):
            b_final[y] = (b_final[y] + best_fitness_R[y])/2
        
        for y in range(len(Av_final)):
            Av_final[y] = (Av_final[y] + fitness_history_R[y])/2

        for y in range(len(M_final)):
            M_final[y] = (M_final[y] + mode_history_R[y])/2
        

    converged_averaged = mean(converged_list)
    # print(converged_averaged)
    graded = [ (fitness(x, target), x) for x in pop_R] #Create a 2D list
    graded = [ x[1] for x in sorted(graded) ]
    # print(graded)

converged_x = list(zip(*converged_list))[1]
converged_y = list(zip(*converged_list))[0]
# print(converged_x)
# print(converged_y)

plt.title("Fitness vs Generations")
plt.xlabel("mutation probability")
plt.ylabel("Average Fitness")
plt.plot(converged_x, converged_y,'tab:orange')
plt.grid()
plt.show()
    
# print(bestInd_allruns)
# print("Target: ", target)
# print("Previous Best Population: ", graded[0])
# print("Best Individual: ", bestInd_allruns[1])

# plt.subplot(221)
# plt.title("Fitness vs Generations")
# plt.xlabel("Generations")
# plt.ylabel("Average Fitness")
# plt.plot(Av_final,'tab:orange')
# plt.grid()


# x_coordList = []
# y_OptimalcoordList = []
# y_GeneratedcoordList = []

# x_coordList = np.linspace(-10,10,100)
# for y in x_coordList:
#     yOValue = poly_comp(y, target)
#     y_OptimalcoordList.append(yOValue)

#     yGValue = poly_comp(y, best_individual)
#     y_GeneratedcoordList.append(yGValue)

# plt.subplot(222)
# plt.title("Polynomial Graph Method A")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.plot(x_coordList, y_OptimalcoordList, 'tab:orange')
# plt.plot(x_coordList, y_GeneratedcoordList, 'tab:purple')
# plt.grid()

# plt.subplot(223)
# plt.title("Modal Plot Method A")
# plt.xlabel("Generations")
# plt.ylabel("Fitness")
# plt.plot(mode_history_R)
# plt.grid()

# plt.subplot(224)                                  
# plt.title("Best Individual Fitness Method A")
# plt.xlabel("Generations")
# plt.ylabel("Fitness")
# plt.plot(b_final, 'tab:green')
# plt.grid()
# plt.show()
