import numpy as np
import matplotlib.pyplot as plt 

# A single perceptron function
def perceptron(inputs_list, weights_list, bias):
    #Convert the inputs list into a np array
    inputs = np.array(inputs_list)

    # Convert the weights list into a np array
    weights = np.array(weights_list)

    # Calculate the dot product
    summed = np.dot(inputs, weights)

    # Add in the bias
    summed = summed + bias

    # Calculate the output
    # N.B. this is a ternary operator
    output = 1 if summed > 0 else 0
    return output

def intercepts_calc(weights_list, b):
    inter_x = -(b / weights_list[0])
    inter_y = -(b / weights_list[1])

    return [inter_x, 0], [0, inter_y]

    

def sim_bool(weights_list, bias, input_range, title):
    fig = plt.xkcd() 

    plt.xlim(-2, 2) 
    plt.ylim(-2, 2) 

    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.title("State Space: " + title)

    plot_colour = ""

    print("Weights: ", weights_list)
    print("Bias: ", bias)
    full_output = []

    for iter1 in range(input_range):
        for iter2 in range(input_range):
            input = [iter1, iter2]
            out = perceptron(input, weights_list, bias)
            plot_colour = "green" if out == 1 else "red"

            plt.scatter(input[0], input[1], s=50, zorder=3, color=plot_colour)
            print("Point of(" + str(input) + ") = " + str(out))
            full_output.append(out)
    intercepts = intercepts_calc(weights_list, bias)
    plt.axline(intercepts[0], intercepts[1])
    plt.grid(True, linewidth=1, linestyle=':')
    plt.tight_layout() 
    plt.show()
    return full_output

def simulate_xor():
    fig = plt.xkcd() 

    plt.xlim(-2, 2) 
    plt.ylim(-2, 2) 

    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.title("State: boolean XOR")

    for x1 in range(2):
        for x2 in range(2):
            input = [x1, x2]
            out = xor_case([x1, x2])
            plot_colour = "green" if out == 1 else "red"
            plt.scatter(input[0], input[1], s=50, zorder=3, color=plot_colour)
            print("Point of:(" + str(input) + ") = " + str(out))

    intercepts_one = intercepts_calc([2.0, 2.0], -1)
    intercepts_two = intercepts_calc([-1.0, -1.0], 1.5)
    plt.axline(intercepts_one[0], intercepts_one[1])
    plt.axline(intercepts_two[0], intercepts_two[1])
    plt.grid(True, linewidth=1, linestyle=':')
    plt.tight_layout() 
    plt.show()

def or_case(inputs_list):
    return perceptron(inputs_list, [2.0, 2.0], -1)

def and_case(inputs_list):
    return perceptron(inputs_list, [1.0, 1.0], -1)

def nor_case(inputs_list):
    return perceptron(inputs_list, [-1.0, -1.0], 1)

def nand_case(inputs_list):
    return perceptron(inputs_list, [-1.0, -1.0], 2)

def xor_case(inputs_list):
    return and_case([or_case(inputs_list), nand_case(inputs_list)])

if __name__ == "__main__":
    print("AND Logic Table:")
    print(sim_bool([1.0, 1.0], bias=-0.5, input_range=2, title="boolean AND"))

    print("NAND Logic Table:")
    print(sim_bool([-1.0, -1.0], bias=1.5, input_range=2, title="boolean NAND"))

    print("OR Logic Table:")
    print(sim_bool([2.0, 2.0], bias=-1, input_range=2, title="boolean OR"))

    print("XOR Logic Table:")
    simulate_xor()