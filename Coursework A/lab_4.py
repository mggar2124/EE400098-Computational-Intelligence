import numpy as np
from matplotlib import pyplot as plt
from ANN import NeuralNetwork
from enum import Enum

S_MNIST_TRAINING_DATA= "MNIST/mnist_train_100.csv"
L_MNIST_TRAINING_DATA= "MNIST/mnist_train.csv"
S_MNIST_TEST_DATA = "MNIST/mnist_test_10.csv"
L_MNIST_TEST_DATA = "MNIST/mnist_test.csv"

S_FASH_MNIST_TRAINING_DATA= "MNIST/fashion_mnist_train_100.csv"
L_FASH_MNIST_TRAINING_DATA= "MNIST/fashion_mnist_train.csv"
S_FASH_MNIST_TEST_DATA = "MNIST/fashion_mnist_test_10.csv"
L_FASH_MNIST_TEST_DATA = "MNIST/fashion_mnist_test.csv"

def train(neural_network: NeuralNetwork, data_source, iterations):
    output_nodes = neural_network.o_nodes
    # Load the MNIST 100 training samples CSV file into a list
    training_data_file=open(data_source, 'r')
    training_data_list=training_data_file.readlines()
    training_data_file.close()

    for i in range(iterations):
        # Train the neural network on each training sample
        for record in training_data_list:
            # Split record by the commas
            all_values = record.split(',')
            
            # Scale and shift the inputs from 0..255 to 0.01..1
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            # Create the target output values (all 0.01, except the desire label which is 0.99)
            targets = np.zeros(output_nodes) + 0.1

            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99 
            neural_network.train(inputs, targets)

def test(neural_network, data_source):
    # Load the MNIST test samples CSV file into a list  
    test_data_file = open(data_source, 'r')  
    test_data_list = test_data_file.readlines()  
    test_data_file.close()  
    
    # Scorecard list for how well the network performs, initially empty  
    scorecard = []  
    
    # Loop through all of the records in the test data set  
    for record in test_data_list:
        # Split the record by the commas  
        all_values = record.split(',') 
        
        # The correct label is the first value  
        correct_label = int(all_values[0])  
        print(correct_label, "Correct label")  
        # Scale and shift the inputs 
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  
        # Query the network 
        outputs = neural_network.query(inputs) 
        
        # The index of the highest value output corresponds to the  label  
        label = np.argmax(outputs) 
        print(label, "Network label") 
        # Append either a 1 or a 0 to the scorecard list  
        if (label == correct_label):
            scorecard.append(1) 
        else:
            scorecard.append(0) 
        pass 
    # Calculate the performance score, the fraction of correct answers 
    scorecard_array = np.asarray(scorecard) 
    print("Scorecard sum =", scorecard_array.sum())
    performance = (scorecard_array.sum() *100.0 / scorecard_array.size)
    print("Performance = ", performance , '%')
    return performance 

def plot_neural_network_test_results(array_of_coordinates, label, x_axis):
    data = np.array(array_of_coordinates)
    x, y = data.T
    plt.xlabel(x_axis)
    plt.title(label)
    plt.ylabel("Network performance (%)")
    plt.plot(x, y)
    plt.show()

def learning_rate_experimentation(starting_rate, repetitions):
    results = []
    lr_range = starting_rate * 20
    for i in range(lr_range):
        current_learning_rate = starting_rate / (i + 1)
        # Take average of repetitions at given learning rate
        av_list = []
        for j in range(repetitions):
            neural_network = NeuralNetwork(input_nodes=784, hidden_nodes=60, output_nodes=10, learning_rate=current_learning_rate)
            train(neural_network, data_source=L_FASH_MNIST_TRAINING_DATA, iterations=1)
            av_list.append(test(neural_network, L_FASH_MNIST_TEST_DATA))
        results.append([current_learning_rate, sum(av_list) / repetitions])
        print(f"Completed iteration {i}/{lr_range}")
    for each in results:
        print(each)
    plot_neural_network_test_results(results, "Fashion MNIST performances changes as a function of learning rate", "Learning Rate")

# Default value is 100
def perceptron_count_experimentation(repetitions):
    results = []    
    for i in range(50, 1000, 50):
        repetitions = repetitions
        av_list = []
        for j in range(repetitions):
            neural_network = NeuralNetwork(input_nodes=784, hidden_nodes=(i + 1), output_nodes=10, learning_rate=0.25)
            train(neural_network, data_source=L_FASH_MNIST_TRAINING_DATA, iterations=1)
            av_list.append(test(neural_network, L_FASH_MNIST_TEST_DATA))
        results.append([i, sum(av_list) / repetitions])
    plot_neural_network_test_results(results, "Performance changes as a function of hidden node count", "Hidden node count")

def train_multiple_times(neural_network, training_dataset, testing_dataset, repetitions):
    results = []
    for i in range(repetitions):
        train(neural_network, training_dataset, 1)
        results.append([i, test(neural_network, testing_dataset)])
    plot_neural_network_test_results(results, "Performance of network after training repetition", "Epoch")
    for each in results: 
        print(each)
    return results
# Optimal for fashion: hidden count = 0.05, node count = 800
if __name__ == "__main__":
    learning_rate_experimentation(starting_rate=2, repetitions=2)
    perceptron_count_experimentation(1)

    nn = NeuralNetwork(input_nodes=784, hidden_nodes=100, output_nodes=10, learning_rate=0.3)
    train(nn, data_source="data/mnist_train_100.csv", iterations=1)
    test(nn, "data/mnist_test_10.csv")
    learning_rate_experimentation(1, 2)

    # Test training cycles
    # nn = NeuralNetwork(784, 250, 10, 0.1)
    # train_multiple_times(nn, L_MNIST_TRAINING_DATA, L_MNIST_TEST_DATA, 50)
    nn = NeuralNetwork(input_nodes=784, hidden_nodes=800, output_nodes=10, learning_rate=0.05)
    train_multiple_times(nn, L_FASH_MNIST_TRAINING_DATA, L_FASH_MNIST_TRAINING_DATA, 20)