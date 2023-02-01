import numpy

def perceptron(inputs_list, weights_list, bias):
    # Convert the inputs list into a numpy array
    inputs = numpy.array(inputs_list)
    # Convert the weights list into a numpy array
    weights = numpy.array(weights_list)
    # Calculate the dot product
    summed = numpy.dot(inputs, weights)
    # Add in the bias
    summed = summed + bias
    # Calculate output
    # N.B this is a ternary operator, neat huh? output = 1 if summed > 0 else 0
    output = 1 if summed > 0 else 0
    return output

W = [1.0, 1.0]
b = -1
X = [0,1]

perceptron(X, W, b)