import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

import numpy as np
import logging

def create_model(hidden_units=25):
    # Set seed for reproducibility
    tf.random.set_seed(7)

    # Create a Sequential model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(
            hidden_units,
            activation='relu',
            kernel_initializer='HeNormal',
            bias_initializer='zeros'
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(
            10,
            activation='softmax',
            kernel_initializer='GlorotUniform',
            bias_initializer='zeros',
        ),
    ])

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

def train_model(hidden_units=25, epochs=10, batch_size=64):
    # Load the dataset
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Standarize the data
    mean = np.mean(train_images)
    stddev = np.std(train_images)
    train_images = (train_images - mean) / stddev
    test_images = (test_images - mean) / stddev

    # One-hot encode the labels
    train_labels = to_categorical(train_labels, num_classes=10)
    test_labels = to_categorical(test_labels, num_classes=10)

    # Create the KerasClassifier
    model = KerasClassifier(build_fn=create_model, hidden_units=hidden_units, epochs=epochs, batch_size=batch_size, verbose=0)

    # Define the grid search parameters
    param_grid = {
        'hidden_units': [25, 50, 100],
        'optimizer': ['adam', 'sgd', 'rmsprop'],
        'activation': ['relu', 'softmax', 'sigmoid'],
        'epochs': [5, 10, 20],
        'batch_size': [32, 64, 128]
    }

    # Create and fit the GridSearchCV
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(train_images, train_labels)

    # Summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

if __name__ == '__main__':
    train_model()