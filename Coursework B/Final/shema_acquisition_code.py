import random 
from operator import add
import matplotlib.pyplot as plt 
from functools import reduce
import numpy as np


def individual(length, min, max):
    return [random.randint(min, max) for x in range(length)]

def population(count, length, min, max):
    return [individual(length, min, max) for x in range(count)]

def fitness(individual, target):
    targetArray = np.array(target); indArray = np.array(individual)
    sum = np.sum(abs(np.subtract(targetArray,indArray)))
    return sum

def grade(population, target):
    summed = reduce(add, (fitness(x, target) for x in population))

    graded = [(fitness(x, target), x) for x in population]
    fit = [x[0] for x in graded]
    for fit in graded:
        if fit == 0:
            fit = 0.0000001
    
    total = sum([(1/x[0]) for x in graded])
    return summed / (len(population) * 1.0)

def evolve(population, target, retain=0.1, random_select=0.05, mutate=0.1):
    graded = [(fitness(x, target), x) for x in population]
    fits = [x[0] for x in graded]
    for fit in fits:
        if fit == 0:
            fit = 0.0000001
    total = sum([(1/x[0]) for x in graded])

    selection_list = []
    weights_list = []
   
    retain_length = retain * len(population)

    for x in range(len(graded)):
        weights_list.append((1/fits[x])/total)
        selection_list.append(x)
    
    parent_selection = np.random.choice(selection_list, int(retain_length), p = weights_list)
    parents = [population[x] for x in parent_selection]
    for individuals in parents:
        if mutate > random.random():
            position_mutated = random.randint(0, len(individuals) - 1)
            individuals[position_mutated] = random.randint(min(individuals), max(individuals))
 
    # Crossover parents to create children
    parents_length = len(parents)
    desired_length = len(population) - parents_length
    
    children = []

    while len(children) < desired_length:
        male = random.randint(0, parents_length - 1)
        female = random.randint(0, parents_length - 1)
        if male != female:
            male = parents[male]
            female = parents[female]
           
            point = random.randint(1, (len(male)-2))      
            child = male[:int(point)] + female[int(point):]
            children.append(child)
    parents.extend(children)


    for individual in parents:
        if mutate > random.random():
            position_mutated = random.randint(0, len(individual)-1) 
            individual[position_mutated] = random.randint(min(individual), max(individual))
    return parents



number = [25, 18, 31, -14, 7 , -19]
target = [ 0 , 1 , 1 , 0 , 0 , 1 ,   0 , 1 , 0 , 0 , 1 , 0 ,      0 , 1 , 1 , 1 , 1 , 1 ,     1 , 0 , 1 , 1 , 1 , 0 ,     0 , 0 , 0 , 1 , 1 , 1 ,     1 , 1 , 0 , 0 , 1 , 1 ]
# Low order High length (4, 32)
schema = ['*','*', 1 ,'*','*','*',  '*','*', 0 ,'*','*','*',     '*','*','*','*','*','*',     1 ,'*','*','*','*','*',    '*','*','*','*','*','*',    '*','*','*', 0 ,'*','*']

# low order low length (3, 3)
# schema = ['*','*', 1 , 0 , 0 ,'*',  '*','*','*','*','*','*',     '*','*','*','*','*','*',    '*','*','*','*','*','*',    '*','*','*','*','*','*',    '*','*','*','*','*','*']

# high order low length (18, 18)
# schema = ['*','*','*','*','*','*',   0 , 1 , 0 , 0 , 1 , 0 ,      0 , 1 , 1 , 1 , 1 , 1 ,     1 , 0 , 1 , 1 , 1 , 0,    '*','*','*','*','*','*',    '*','*','*','*','*','*']

# high order high length (12, 36)
# schema = [0 , 1 , 1 , 0 , 0 , 1 ,  '*','*','*','*','*','*',     '*','*','*','*','*','*',    '*','*','*','*','*','*',    '*','*','*','*','*','*',     1 , 1 , 0 , 0 , 1 , 1]


# schema = ['*','*','*','*','*','*',  '*','*','*','*','*','*',     '*','*','*','*','*','*',    '*','*','*','*','*','*',    '*','*','*','*','*','*',    '*','*','*','*','*','*']

population_count = 300
inidivual_length = 36
individual_min = 0
inidividual_max = 1
generations =  100

schema_fitness = []
schema_probability = []
schema_chromosomes = []
schema_count = []
population_average = []

p = population(population_count, inidivual_length, individual_min, inidividual_max)
fitness_history = [grade(p, target)]

for i in range(generations):
    schema_chromosomes = []
    count = 0
    p = evolve(p, target)
    next_grade = grade(p, target)
    fitness_history.append(next_grade)

    graded = [(fitness(x, target), x) for x in p]
    graded = [ x[1] for x in sorted(graded)]

    for chromosomes in p:
        for x in range(len(chromosomes)):
            if (schema[x] != '*') and (schema[x] != chromosomes[x]):
                break
            elif x == 35:
                schema_chromosomes.append(chromosomes)
                count += 1
    schema_fitness.append(np.average([fitness(x, target) for x in schema_chromosomes]))
    population_average.append(np.average([fitness(x, target) for x in p]))

    schema_probability.append((len(schema_chromosomes) * schema_fitness[i] * 100) / (len(p) * population_average[i]))

    if graded[0] == target or i == 10:
        break 

    schema_count.append(count)
 
print("Schema: ", schema)
print("Target: ", target)

print("Number of Schema in Generations: ", schema_count)
print("Average Schema Fitness for each Generation: ", schema_fitness)
print("Average Population Fitness for each Generation: ", population_average)

order = 4
definition_length = 32

cross_C = (1-((0.9*17)/35))
mutate_C = (1-0.05)**order

expected_count = []

for i in range(len(schema_count)):
        expected_count.append(schema_count[i]*((36-schema_fitness[i])/(36-population_average[i]))*cross_C*mutate_C)


print("Overestimated Count for Next Generation: ", expected_count)
