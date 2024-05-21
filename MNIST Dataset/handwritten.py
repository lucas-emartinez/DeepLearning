import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

TRAIN_IMAGE_FILENAME = './train-images.idx3-ubyte'
TRAIN_LABEL_FILENAME = './train-labels.idx1-ubyte'
TEST_IMAGE_FILENAME = './t10k-images.idx3-ubyte'
TEST_LABEL_FILENAME = './t10k-labels.idx1-ubyte'

def read_mnist():
    train_images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)
    train_labels = idx2numpy.convert_from_file(TRAIN_LABEL_FILENAME)
    test_images = idx2numpy.convert_from_file(TEST_IMAGE_FILENAME)
    test_labels = idx2numpy.convert_from_file(TEST_LABEL_FILENAME)

    np.random.seed(7)  # To make repeatable
    LEARNING_RATE = 0.01
    EPOCHS = 20

    # Reformat and standardize.
    x_train = train_images.reshape(60000, 784)
    mean = np.mean(x_train)
    stddev = np.std(x_train)
    x_train = (x_train - mean) / stddev
    x_test = test_images.reshape(10000, 784)
    x_test = (x_test - mean) / stddev

    # One-hot encoded output.
    y_train = np.zeros((60000, 10))
    y_test = np.zeros((10000, 10))

    for i, y in enumerate(train_labels):
        y_train[i][y] = 1
    for i, y in enumerate(test_labels):
        y_test[i][y] = 1
    return x_train, y_train, x_test, y_test
    # Read train and test examples.
    x_train, y_train, x_test, y_test = read_mnist()
    index_list = list(range(len(x_train)))  # Used for random order


# Print dimensions.
# print('dimensions of train_images: ', train_images.shape)
# print('dimensions of train_labels: ', train_labels.shape)
# print('dimensions of test_images: ', test_images.shape)
# print('dimensions of test_images: ', test_labels.shape)


# # Print one training example.
# print('label for first training example: ', train_labels[0])
# print('---beginning of pattern for first training example---')
# for line in train_images[0]:
#     for num in line:
#         if num > 0 :
#             print('*', end = ' ')
#         else:
#             print(' ', end = ' ')
#     print(' ')
# print('---end of pattern for first training example---')

# Read train and test examples.
x_train, y_train, x_test, y_test = read_mnist()
index_list = list(range(len(x_train))) # Used for random order


def layer_w(neuron_count, input_count):
    weights = np.zeros((neuron_count, input_count+1))
    for i in range(neuron_count):
        for j in range(1, (input_count+1)):
            weights[i][j] = np.random.uniform(-0.1, 0.1)
    return weights