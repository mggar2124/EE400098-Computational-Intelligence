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


def individual(length, min, max):
    "Create a member of the population."
    return [ randint(min,max) for x in range(length)]

def population(count, length, min, max):
    return [ individual(length, min, max) for x in range(count) ]

def polynomial_comp(x, coeffecient):
    print(coeffecient)
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
    targets_array = np.array(target); individuals_array = np.array(individual)
    summed = np.sum(abs(np.subtract(targets_array,individuals_array)))
    return summed             
def grade(population, target, best_fitness, best_individual):
    'Find average fitness for a population.'
    fitness_list = []
    for x in range(len(population)):
        individual_fitness = fitness(population[x], target)                             
        if individual_fitness <  individual_fitness or best_fitness == 0:                       
            best_fitness = individual_fitness
            best_individual = population[x]
        fitness_list.append(individual_fitness)
    
    summed = (reduce(add, fitness_list))/(len(population) * 1.0)
    list_mode = mode(fitness_list)
    
    return summed, best_individual, best_fitness, list_mode

def evolve(pop, target, population_fitness, evolve_type, elitism, crossover_type, retain=0.1, random_select=0.05, mutate1=0.3, mutate2=0.3):
    
    graded = [ (fitness(x, target), x) for x in pop] 
    
    if evolve_type == 'roullete': #ROULLETTE

        inverted_fitness = [ 1/(0.1+i[0]) for i in graded ]

        retain_length = int(len(graded)*retain)
        parents = []

        summed = sum(inverted_fitness)
        fitness_roullete = [ i/summed for i in inverted_fitness ]
        probability_distribution = [sum(fitness_roullete[:i+1]) for i in range(len(fitness_roullete))]


        for i in range(retain_length):
            selector = random()
            for (i,individual) in enumerate(pop):
                if selector <= probability_distribution[i]:
                    parents.append(individual)
                    break
    
    elif evolve_type == 'ranked': # RANKED

        graded = [ x[1] for x in sorted(graded) ]
        retain_length = int(len(graded)*retain)
        parents = graded[:retain_length]
        for individual in graded[retain_length:]:
            if random_select > random():
                parents.append(individual)




    parents_length = len(parents)
    desired_length = len(pop) - parents_length 

    children = []

    while len(children) < desired_length:
        male = randint(0, parents_length-1) 
        female = randint(0, parents_length-1) 

        if male != female: 
            male = parents[male]
            female = parents[female]

            ####### CROSS OVER #######
            if crossover_type == "1":
                ### 1 Point cross-over
                point = randint(1, (len(male)-2))      
                child = male[:int(point)] + female[int(point):] 
                children.append(child)      
            elif crossover_type == "2":
                ### 2 Point cross-over
                third_point = int(len(male)/3)
                point1 = randint(1, third_point); point2 = randint(point1,(len(male)-1)) 
                child = male[:point1] + female[point1:point2] + male[point2:]
                children.append(child) 
            elif crossover_type == "uniform":
                ### Uniform cross-over
                child = []
                for x in range(len(male)):
                    selector = randint(0,1)
                    if selector == 0:
                        child.append(male[x])
                    elif selector == 1:
                        child.append(female[x])
                    children.append(child)

    parents.extend(children) 
    n = 0
    if elitism == 1:
        # calculate number of elite members to pass onto the next generation
        n = int(0.05*len(parents))
        # select elite members
        elite = parents[:n]
        parents = parents[n:]


    for individual in parents:
        if fitness(individual, target) > population_fitness:
            if mutate1 > random():
                pos_to_mutate = randint(0, len(individual)-1) #Chooses a random number within individual list
                individual[pos_to_mutate] = randint(min(individual), max(individual))

        elif fitness(individual, target) < population_fitness:
            if mutate2 > random():
                pos_to_mutate = randint(0, len(individual)-1) #Chooses a random number within individual list
                individual[pos_to_mutate] = randint(min(individual), max(individual)) 

    if elitism == 1:
        parents = elite + parents   

    return parents

target = [-19, 7, -14, 31, 18, 25]

population_count = 500
individual_length = 6
individual_min = -35
individual_max = 35
generations = 100 

p_original = population(population_count, individual_length , individual_min , individual_max )

best_individual_history = []
mode_history = []

best_individual = []
best_fitness = 5e+30 #picked
fitness_history = []

grade_population = []
seeded_fitness_history = []


evolution_type = 'ranked'
elitism = 0
final_average = []
final_modal = []
final_best = []
best_individual = []
best_fitness = 5e+30

best_individuals_allruns = []

graded = [(fitness(x, target), x) for x in p_original] 
graded = [x[1] for x in sorted(graded)]
best_individuals_allruns = [fitness(graded[0], target),graded[0]]


for x in range(5):
    print(x)
    population_R = p_original
    fitness_history_R = []
    mode_history_R = []
    best_fitness_R = []
    population_fitness, best_individual, best_fitness, fitness_mode = grade( p_original , target, best_fitness, best_individual)
    fitness_history_R.append(population_fitness)
    mode_history_R.append(fitness_mode)
    best_fitness_R.append(best_fitness)

    for y in range(generations):
        population_R = evolve( population_R , target, fitness_history_R[y-1], evolution_type, elitism, '1')
        population_fitness, best_individual, best_fitness, fitness_mode = grade( population_R, target, best_fitness, best_individual)
        fitness_history_R.append(population_fitness); mode_history_R.append(fitness_mode); best_fitness_R.append(best_fitness)
    
    if best_fitness <= best_individuals_allruns[0]:
        best_individuals_allruns = [best_fitness, best_individual]

    if x == 0:
        final_average = fitness_history_R

        final_best = best_fitness_R

        final_modal = mode_history_R
    else:
        for y in range(len(final_average)):
            final_average[y] = (final_average[y] + fitness_history_R[y])/2

        for y in range(len(final_best)):
            final_best[y] = (final_best[y] + best_fitness_R[y])/2

        for y in range(len(final_modal)):
            final_modal[y] = (final_modal[y] + mode_history_R[y])/2


graded = [ (fitness(x, target), x) for x in population_R] #
graded = [ x[1] for x in sorted(graded) ]


print(best_individuals_allruns)
print("Target: ", target)
print("Previous Best Population: ", graded[0])
print("Best Individual: ", best_individuals_allruns[1])

plt.subplot(221)
plt.title("Fitness vs Generations")
plt.xlabel("Generations")
plt.ylabel("Average Fitness")
plt.plot(final_average,'tab:orange')
plt.grid()


x_coordinate_list = []
y_optimal = []
y_generated_coordinates = []

x_coordinate_list = np.linspace(-10,10,100)
for y in x_coordinate_list:
    y_value = polynomial_comp(y, target)
    y_optimal.append(y_value)

    y_generated_value = polynomial_comp(y, best_individual)
    y_generated_coordinates.append(y_generated_value)

plt.subplot(222)
plt.title("Polynomial Graph Method A")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x_coordinate_list, y_optimal, 'tab:orange')
plt.plot(x_coordinate_list, y_generated_coordinates, 'tab:purple')
plt.grid()

plt.subplot(223)
plt.title("Modal Plot Method A")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.plot(mode_history_R)
plt.grid()

plt.subplot(224)                                  
plt.title("Best Individual Fitness Method A")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.plot(final_best, 'tab:green')
plt.grid()
plt.show()
