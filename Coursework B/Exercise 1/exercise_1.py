from random import randint, random
from operator import add
import matplotlib.pyplot as plt
from functools import reduce

def individual(length, min, max):
    "Create a member of the population."
    return [ randint(min,max) for x in range(length)]

def population(count, length, min, max):
    """
        Create a number of individuals (i.e. a population).

        count: the number of individuals in the population
        length: the number of values per individual
        min: the minimum possible value in an individual's list of values
        max: the maximum possible value in an individual's list of values

    """
    return [ individual(length, min, max) for x in range(count) ]

def fitness(individual, target):
    """
        Determine the fitness of an individual. Lower is better. (For this function)

        individual: the individual to evaluate
        target: the target number individuals are aiming for
    """
    individualInt = int("".join(str(x) for x in individual), 2) 
    return abs(target-individualInt) 

def grade(pop, target):
    'Find average fitness for a population.'
    summed = reduce(add, (fitness(x, target) for x in pop)) 
    return summed / (len(pop) * 1.0)

def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01):
    graded = [ (fitness(x, target), x) for x in pop] 
    graded = [ x[1] for x in sorted(graded)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)



    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1) 

            individual[pos_to_mutate] = randint(min(individual), max(individual)) 

    parents_length = len(parents)
    desired_length = len(pop) - parents_length 
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1) 
        female = randint(0, parents_length-1) 
        if male != female: 
            male = parents[male]
            female = parents[female]

            ### 1 Point crossover
            half = len(male) / 2       
            child = male[:int(half)] + female[int(half):]
            children.append(child)
    
    parents.extend(children) 
    return parents

#MAIN CODE
target = 550 #1023 max
population_count = 250
i_length = 10
i_min = 0
i_max = 1
generations = 100 #iterations
p = population(population_count, i_length , i_min , i_max )  

fitness_history = [grade(p, target),]
for i in range(generations):
    p = evolve(p, target)
    fitness_history.append(grade(p, target))

for datum in fitness_history:
    print(datum)

plt.plot(fitness_history)
plt.show()
