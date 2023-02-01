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

def outlier_regector(datum, n=2):
    return datum[abs(datum - np.mean(datum)) < n * np.std(datum)]

def individual(length, min, max):
    "Create a member of the population."
    return [ randint(min,max) for x in range(length)]

def population(count, length, min, max):
    return [ individual(length, min, max) for x in range(count) ]

def polynomial_comp(x, coeffecient):
    result = coeffecient[0]
    for i in range(1, len(coeffecient)):
        result = result + coeffecient[i] * x**i
    return result

def fitness(individual, y_coordinate, x_coordinate):
    y_fitness = []
    for x in range(len(x_coordinate)):                        #Loop through x coordinates, calculate polynomial y for each, find fitness to nominal y
        y_output = polynomial_comp(x_coordinate[x], individual)     #Calculate y for individual set of coefficients
        y_fitness.append(abs(y_coordinate[x] - y_output))     #Append list of fitnesses for all coordinate pairs by the difference
        summed = reduce(add, y_fitness)                  
    return summed / (len(y_fitness) * 1.0)             #Return an average fitness for all coordinates using individual's genes

def grade(population, y_coordinate, x_cooridinate, best_fitness, best_individual):
    'Find average fitness for a population.'
    seeded_individual_fitness = []
    fitness_list = []

    for x in range(len(population)):
        individual_fitness = fitness(population[x], y_coordinate, x_cooridinate)                             #Calculate fitness of individual in population
        if individual_fitness <= best_fitness or individual_fitness == 0:                        #Record the best fitness and individual in population
            best_fitness = individual_fitness; best_individual = population[x]
        fitness_list.append(individual_fitness)
        if x < 0.05*len(population):                                            #Code to make an average of only the top results
            seeded_individual_fitness.append(individual_fitness)
    
    fitness_list = outlier_regector(np.array(fitness_list), 2)
    summed = (reduce(add, fitness_list))/(len(population) * 1.0)
    seeded_summed = (reduce(add, seeded_individual_fitness))/(len(seeded_individual_fitness) * 1.0)
    list_mode = mode(fitness_list)
    highest_fitness = min(fitness_list)
    
    return summed, seeded_summed, best_individual, best_fitness, list_mode, highest_fitness

def evolve(pop, y_coordinate, x_coordinate, populationFitness, evolve_type, elitism, crossover, retain=0.2, random_select=0.05, mutate1=0.6, mutate2=0.3): #Mutate 1: shitty ones
    
    graded = [ (fitness(x, y_coordinate, x_coordinate), x) for x in pop] #Create a 2D list
    #####
    if evolve_type == 1:
        inverted_fitness = [ 1/(0.05+i[0]) for i in graded ]
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

    elif evolve_type == 0: # RANKED

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

            if crossover == 1:
                ### 1 Point cross-over
                point = randint(1, (len(male)-2))      
                child = male[:int(point)] + female[int(point):] 
                children.append(child)      
            elif crossover == 2:
                ### 2 Point cross-over
                third_point = int(len(male)/3)
                point1 = randint(1, third_point); point2 = randint(point1,(len(male)-1)) 
                child = male[:point1] + female[point1:point2] + male[point2:]
                children.append(child) 
            elif crossover == 3:
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

    # only use elitism if selection is done through ranking
    if elitism == 1:
        n = int(0.05*len(parents))
        elite = parents[:n]
        parents = parents[n:]

    for individual in parents:

        if fitness(individual, y_coordinate, x_coordinate) >= populationFitness:
            if mutate1 > random():
                pos_to_mutate = randint(0, len(individual)-1) #Chooses a random number within individual list
                individual[pos_to_mutate] = randint(min(individual), max(individual))
        elif fitness(individual, y_coordinate, x_coordinate) < populationFitness:
            if mutate2 > random():
                pos_to_mutate = randint(0, len(individual)-1) #Chooses a random number within individual list
                individual[pos_to_mutate] = randint(min(individual), max(individual)) 

    if elitism == 1:
        parents = elite + parents    

    return parents

### MAIN CODE ###
target = [-19, 7, -14, 31, 18, 25]

population_count = 300
individual_length = 6
individual_min = -100
individual_max = 100
generations = 1000 

p_original = population(population_count, individual_length , individual_min , individual_max )

x_coordinate_list = []
y_coordinate_list = []
x_coordinate_list = np.linspace(-20,20,10)
for z in x_coordinate_list:
    y_value = polynomial_comp(z, target)
    y_coordinate_list.append(y_value)

graded = [ (fitness(x, y_coordinate_list, x_coordinate_list), x) for x in p_original] #Create a 2D list
graded = [ x[1] for x in sorted(graded) ]
best_individuals_allruns = [fitness(graded[0], y_coordinate_list, x_coordinate_list),graded[0]]

print("Beginning")

evovle_type = 0
elitism = 0
final_average = []
final_modal = []
final_best = []
best_individual = []
best_fitness = 5e+30

start_value = 1
end_value = 0
iterations = 50

converged_list = []
converged_averaged = []

for x in range(iterations):
    varying_parameter = start_value - start_value*x/iterations
    mutation = varying_parameter
    print(varying_parameter)
    pRU = p_original
    fitness_history_RU = []
    
    best_fitness_R = []
    mode_history_R = []

    population_fitness, seeded_fitness, best_individual, best_fitness, fitness_modal, highest_fitness = grade( p_original , y_coordinate_list, x_coordinate_list, best_fitness, best_individual)
    fitness_history_RU.append(population_fitness)

    
    stop = 1
    for y in range(generations):
        pRU = evolve( pRU , y_coordinate_list, x_coordinate_list, fitness_history_RU[y-1], evovle_type, elitism, 1, mutate2=mutation)
        population_fitness, seeded_fitness, best_individual, best_fitness, fitness_modal, highest_fitness = grade( pRU, y_coordinate_list, x_coordinate_list, best_fitness, best_individual)
        # print(fitness_modal)

        if highest_fitness <= 10 and stop != 0:
            # print("arrived")
            converged_list.append([y, varying_parameter])
            print(y, highest_fitness)
            stop = 0

        fitness_history_RU.append(population_fitness); mode_history_R.append(fitness_modal); best_fitness_R.append(best_fitness)


    if best_fitness <= best_individuals_allruns[0]:
        best_individuals_allruns = [best_fitness, best_individual]

    if x == 0:
        final_average = fitness_history_RU
    else:
        for z in range(len(final_average)):
            final_average[z] = (final_average[z] + fitness_history_RU[z])/2

    if x == 0:
        final_modal = mode_history_R
    else:
        for z in range(len(final_modal)):
            final_modal[z] = (final_modal[z] + mode_history_R[z])/2

    if x == 0:
        final_best = best_fitness_R
    else:
        for z in range(len(final_best)):
            final_best[z] = (final_best[z] + best_fitness_R[z])/2

graded = [ (fitness(x, y_coordinate_list, x_coordinate_list), x) for x in pRU] #Create a 2D list
graded = [ x[1] for x in sorted(graded) ]
print(best_individuals_allruns)
print("Targets: ", target)
print("Previous Population's Best: ", graded[0])
print("Best Individual: ", best_individuals_allruns[1])

converged_x = list(zip(*converged_list))[1]
converged_y = list(zip(*converged_list))[0]
print(converged_x)
print(converged_y)

plt.title("Mutations vs Generations to Reach Acceptable Fitnesss")
plt.xlabel("Mutation Probability")
plt.ylabel("Generations to Reach Acceptable Fitnesss")
plt.plot(converged_x, converged_y,'tab:blue')
plt.grid()
plt.show()



# y_generated_coordinates = []
# for z in x_coordinate_list:
#     y_value = polynomial_comp(z, best_individual)
#     y_generated_coordinates.append(y_value)

# plt.subplot(221)
# plt.title("Fitness vs Generations Method A")
# plt.xlabel("Generations")
# plt.ylabel("Average Fitness")
# plt.plot(final_average,'tab:red')
# plt.grid()
# plt.show()

# plt.subplot(222)
# plt.title("Polynomial Graph Method A")
# plt.xlabel("X axis")
# plt.ylabel("Y axis")
# plt.plot(x_coordinate_list, y_coordinate_list, 'tab:orange')
# plt.plot(x_coordinate_list, y_generated_coordinates, 'tab:purple')
# plt.grid()

# plt.subplot(223)
# plt.title("Modal Plot Method B")
# plt.xlabel("Generations")
# plt.ylabel("Fitness")
# plt.plot(mode_history_R)
# plt.grid()

# plt.subplot(224)                                  
# plt.title("Best Individual Fitness Method B")
# plt.xlabel("Generations")
# plt.ylabel("Fitness")
# plt.plot(final_best, 'tab:green')
# plt.grid()
# plt.show()
