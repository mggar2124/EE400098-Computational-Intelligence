from unicodedata import decimal
import numpy as np
from numpy.core.fromnumeric import repeat
from ANN import NeuralNetwork

#can converted into a sweep
input_nodes = int(input("how many input nodes would you like \n")) #good example is 2
hidden_nodes = int(input("how many hiddne nodes would you like \n")) #good example is 5
output_nodes = int(input("how many output nodes would you like \n")) #good example is 1
learning_rate = float(input("what learning rate would you like \n")) #good example is 0.1

experimental_parameters = [input_nodes, hidden_nodes, output_nodes, learning_rate]


def iterate(neural_network, iterations, training_vectors, target_vectors):
    response = input("to see progress of all iterations enter 1 else enter any other value ")
    for a in range(iterations):
        for b in range(len(training_vectors)):
            neural_network.train(training_vectors[b], target_vectors[b])

        current_output = neural_network.query_multiple(training_vectors)
        if response == "1":
            print(f"Iteration {a + 1}/{iterations}")

    
    print("All iterations have been run")
    return current_output



if __name__ == "__main__":

    test_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    training_vectors = []
    for x0 in range(2):
        for x1 in range(2):
                training_vectors.append([x0, x1])

    and_target_list = [0, 0, 0, 1]
    or_target_list = [0, 1, 1, 1]
    nor_target_list = [1, 0, 0, 0]

   
    print(f"AND Output: {iterate(test_network, 1000, training_vectors, and_target_list)}")
    print(f"OR Output: {iterate(test_network, 1000, training_vectors, or_target_list)}")    
    print(f"NOR Output: {iterate(test_network, 1000, training_vectors, nor_target_list)}")