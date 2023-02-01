from random import randint, random
from operator import add
from statistics import mean
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
population_count = 350
retain_count = 1
individual_length = 10
individual_min = 0
individual_max = 1
generations = 100 
iterations = 10
step_size = 0.01
# p = population(population_count, i_length , i_min , i_max )  
fitness_history = []
generations_of_retain = []
retain_list = []
generations_of_retain_average = []
unconverged_list = []
retain_list_unconverged = []

for x in range (600):
    unconverged_counter = 0
    for i in range (iterations):    
        counter = 0
        generation_counter = 0
        p = population(population_count, individual_length , individual_min , individual_max )  
        current_fitness = grade(p,target)
        fitness_history.append(current_fitness)
        while counter <= 2000:
            p = evolve(p, target, retain= retain_count)
            current_fitness = grade(p, target)
            fitness_history.append(current_fitness)
            counter +=1
            generation_counter +=1
            if counter > 2000:
                unconverged_counter +=1
                # print("unconverged", unconverged_counter)
            if current_fitness <= 0:
                break
        if counter < 2000:
            generations_of_retain.append(generation_counter)

        current_fitness =0
        fitness_history = []
        p = []
        print("number of genrations to completion: ", generation_counter)
        if i == iterations - 1:
            # print the number of unconverged items so it can be plotted later
            print(generations_of_retain)
            print(unconverged_counter)
            if len(generations_of_retain) != 0:
                print("mean: ", mean(generations_of_retain))
                generations_of_retain_average.append(mean(generations_of_retain))
                retain_list.append(retain_count)
                generations_of_retain = []

            if unconverged_counter != 0:
                retain_list_unconverged.append(retain_count)
                unconverged_list.append(unconverged_counter)

    retain_count = retain_count - step_size
    if retain_count <=0:
        break
print(retain_list)
plt.title("Retain Percentage vs Average Generations to Fitness of 0")
plt.xlabel("Retain Percentage")
plt.ylabel("Average Generations to Fitness of 0")
plt.plot(retain_list, generations_of_retain_average)
plt.show()

plt.title("Retain Percentage vs Unconverged Runs")
plt.xlabel("Retain Percentage")
plt.ylabel("Number of Unconverged Runs within Threshold")
print(unconverged_list)
plt.plot( retain_list_unconverged, unconverged_list )
plt.show()
