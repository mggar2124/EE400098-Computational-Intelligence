# Fixed generation termination
fitness_history = [grade( p , target) , ]
for i in range(generations) :
    p = evolve( p, target )
    fitness_history.append(grade( p, target))

# Threshold value termination
count = 0
currentFitness = grade(p,target)
fitness_history = [currentFitness , ]
while count <= 10:
    if currentFitness <= 0.1:
        count += 1
    p = evolve( p, target )
    currentFitness = grade(p,target)
    fitness_history.append(currentFitness)

# stagnation 
count = 0
currentFitness = grade(p,target)
fitness_history = [currentFitness , ]
while count <= 10:
    p = evolve( p, target )
    currentFitness = grade(p,target)
    fitness_history.append(currentFitness)
    difference = abs(fitness_history[len(fitness_history)-2] 
        - fitness_history[len(fitness_history)-1])
    
    if difference <= 0.2:
        count += 1


