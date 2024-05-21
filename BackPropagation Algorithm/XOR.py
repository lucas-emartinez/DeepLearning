

import numpy as np

np.random.seed(3)  # To make repeatable, this makes the random numbers the same each time the code is run

LEARNING_RATE = 0.09
index_list = [0, 1, 2, 3]  # Used to randomize order

# Define training examples
x_train = np.array([
    [1.0, -1.0, -1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, -1.0],
    [1.0, 1.0, 1.0]
])

y_train = [0.0, 1.0, 1.0, 0.0]  # Output (Ground Truth) Between 0 and 1 because we are using a sigmoid function


# Now we declare variables to hold the state our three neurons
# Where input_count is the number of inputs to the neuron
def neuron_w(input_count):
    weights = np.zeros(input_count + 1)  # We add 1 to the input_count to account for the bias

    # We initialize the weights to random values between -1 and 1
    for i in range(1, input_count + 1):
        weights[i] = np.random.uniform(-1.0, 1.0)
    return weights


# We declare the weights for the three neurons
n_w = [neuron_w(2), neuron_w(2), neuron_w(2)]  # We have 2 inputs
n_y = [0.0, 0.0, 0.0]  # We declare the output of the three neurons
n_error = [0.0, 0.0, 0.0]  # Error Term of the three neurons


# These are all the state variables that we need for each neuron for both the
# forward pass and the backward pass: weights (n_w), output (n_y),3 and error
# term (n_error). We arbitrarily initialize the input weights to random numbers
# between −1.0 and 1.0, and we set the bias weights to 0.0. The reason to randomly
# initialize the input weights is to break the symmetry. If all neurons start with the
# same initial weights, then the initial output of all neurons in a layer would also be
# identical. This in turn would lead to all neurons in the layer behaving the same
# during backpropagation, and they would all get the same weight adjustments.

# That is, we do not get any benefit from having multiple neurons in a layer. The
# bias weight does not need to be randomly initialized because it is sufficient to
# randomize the regular input weights to break the symmetry.

# Initializing bias weights to 0.0 is a common strategy.


# Now we make a function to show the learning
def show_learning():
    print('Current weights:')
    for i, w in enumerate(n_w):
        print(
            'Neuron', i,
            ': w0 =', '%5.2f' % w[0],
            ', w1=', '%5.2f' % w[1],
            ', w2=', '%5.2f' % w[2]
        )
    print('----------------')


# The forward_pass function first computes the outputs of neurons 0 and 1 with the same inputs
# (the inputs from the training example) and then puts their outputs into an array,
# together with a bias value of 1.0, to use as input to neuron 2. That is, this function
# defines the topology of the network. We use tanh for the neurons in the first layer
# and the logistic sigmoid function for the output neuron.

def forward_pass(x):
    global n_y  # Global is to make the variable accessible outside the function
    n_y[0] = np.tanh(np.dot(n_w[0], x))  # We use the tanh activation function
    n_y[1] = np.tanh(np.dot(n_w[1], x))  # We use the tanh activation function
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])  # We add 1.0 to account for the bias
    z2 = np.dot(n_w[2], n2_inputs)
    n_y[2] = 1.0 / (1.0 + np.exp(-z2))  # We use the sigmoid activation function


# The backward_pass function starts by computing the derivative of the error
# function and then computes the derivative of the activation function for the output
# neuron.

# The error term of the output neuron is computed by multiplying these two together.

# We then continue to backpropagate the error to each of the two neurons in the hidden layer.
# This is done by computing the derivatives of their activation functions and multiplying these derivatives
# by the error term from the output neuron and by the weight to the output neuron.

def backward_pass(y_truth):
    global n_error
    error_prime = -(y_truth - n_y[2])  # We compute the derivative of the loss-function
    derivative = n_y[2] * (1.0 - n_y[2])  # We compute the derivative of the sigmoid function
    n_error[2] = error_prime * derivative  # We compute the error term for the output neuron
    derivative = 1.0 - n_y[0] ** 2  # We compute the derivative of the tanh function
    n_error[0] = n_w[2][1] * n_error[2] * derivative
    derivative = 1.0 - n_y[1] ** 2
    n_error[1] = n_w[2][2] * error_prime * derivative


# Finally, the adjust_weights function adjusts the weights for each of the three neurons.
# The adjustment factor is computed by multiplying the input by the learning rate and the error term for the neuron
# in question.

def adjust_weights(x):
    global n_w
    n_w[0] -= (LEARNING_RATE * n_error[0] * x)
    n_w[1] -= (LEARNING_RATE * n_error[1] * x)
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])
    n_w[2] -= (LEARNING_RATE * n_error[2] * n2_inputs)


# Network training loop.
all_correct = False
while not all_correct:  # Train until converged
    all_correct = True
    np.random.shuffle(index_list)  # Randomize order
    for i in index_list:  # Train on all examples
        forward_pass(x_train[i])
        backward_pass(y_train[i])
        adjust_weights(x_train[i])
        show_learning()  # Show updated weights
    for i in range(len(x_train)):  # Check if converged
        forward_pass(x_train[i])
        print('x1 =', '%4.1f' % x_train[i][1],
              ', x2 =', '%4.1f' % x_train[i][2],
              ', y =', '%.4f' % n_y[2])
        if ((y_train[i] < 0.5) and (n_y[2] >= 0.5)) or ((y_train[i] >= 0.5) and (n_y[2] < 0.5)):
            all_correct = False

# We pick training examples in random order, call the functions forward_pass,
# backward_pass, and adjust_weights, and then print out the weights with
# the function show_learning. We adjust the weights regardless whether the
# network predicts correctly or not. Once we have looped through all four training
# examples, we check whether the network can predict them all correctly, and if
# not, we do another pass over them in random order.
# We want to point out a couple of issues before running the program. First, you
# might get a different result than our example produces given that the weights are
# initialized randomly. Similarly, there is no guarantee that the learning algorithm
# for a multilevel network will ever converge, and there are multiple reasons for
# this. It could be that the network itself simply cannot learn the function, as we saw
# in Chapter 2 when trying to learn XOR with a single perceptron. Another reason
# convergence might fail is if the parameters and initial values for the learning
# algorithm are initialized in a way that somehow prevents the network from
# learning. That is, you might need to tweak the learning rate and initial weights to
# make the network learn the solution.


