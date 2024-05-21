import random
import matplotlib.pyplot as plt

# First element in the vector x must be 1
# length of w and x must be n+1 for neuron with n inputs
def computeOutput(w, x):
    z = 0.0
    
    for i in range(len(w)):
        z += x[i] * w[i] # Compute the sum of weighted inputs
    
    if z < 0: # Apply the sign Function (Signum)
        return -1
    else:
        return 1

    

# Define variables needed to control training process
random.seed(7) # To make repeatable
LEARNING_RATE = 0.1
index_list = [0, 1, 2, 3] # Used to randomize order

# Define training examples

# First value = 1.0 because is the BIAS
x_train = [(1.0, -1,0 , -1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0), (1.0, 1.0, 1.0)]

y_train =  [1.0, 1.0, 1.0, -1.0]

# Define perceptron weights

w = [0.2, -0.6, 0.25] # Initialize to some "random" numbers

# Print initial weights

# Define variables needed for plotting
color_list = ['r-', 'm-', 'y-', 'c-', 'b-', 'g-']
color_index = 0
def show_learning(w):
    global color_index
    print('w0 =', '%5.2f' % w[0], ', w1=', '%5.2f' % w[1], ', w2=', '%5.2f' % w[2]) 
    if color_index == 0:
        plt.plot([1.0], [1.0], 'b_', markersize=12)
        plt.plot([-1.0, 1.0, -1.0], [1.0, -1.0, -1.0], 'r+', markersize=12)
        plt.axis([-2, 2, -2, 2])
        plt.xlabel('x1')
        plt.ylabel('x2')
    x = [-2.0, 2.0]
    if abs(w[2]) < 1e-5:
        y = [-w[1]/(1e5)*(2.0)+(-w[0]/(1e5)), -w[1]/(1e5)*(2.0)+(-w[0]/(1e5))]
    else:
        y = [(-w[1]/w[2]*(-2.0))+(-w[0]/w[2]), (-w[1]/w[2]*(2.0))+(-w[0]/w[2])]
    plt.plot(x, y, color_list[color_index])
    if color_index < len(color_list)-1:
        color_index += 1



show_learning(w)

# Train the perceptron

all_correct = False
while not all_correct:
    all_correct = True
    random.shuffle(index_list)
    for i in index_list:
        x = x_train[i]
        y = y_train[i]
        p_out = computeOutput(w, x_train[i])
        if y != p_out:
            all_correct = False
            for j in range(len(w)):
                w[j] += (y * LEARNING_RATE  * x[j])
            all_correct = False
            show_learning(w)
     
plt.show()