import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import logging



def train():
    tf.get_logger().setLevel(logging.ERROR)
    tf.random.set_seed(7)

    EPOCHS = 40
    BATCH_SIZE = 32
    # Load the dataset.
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Standarize the data
    mean = np.mean(train_images)
    stddev = np.std(train_images)
    train_images = (train_images - mean) / stddev
    test_images = (test_images - mean) / stddev

    # One-hot encode the labels.
    train_labels = to_categorical(train_labels, num_classes=10)
    test_labels = to_categorical(test_labels, num_classes=10)

    # Initialize the weights.
    #initializer = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)

    # Create a Sequential model.
    # 784 inputs
    # Two Dense (fully connected) layers with 25 and 10 neurons.
    # Tanh activation function for the hidden layer.
    # Logistic (sigmoid) activation function for the output layer.

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(
            25,
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

    # Use stochastic gradient descent (SGD) with
    # learning rate of 0.01 and no other bells and whistles.
    # MSE as loss function and report accuracy during training.
    #opt = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0,nesterov=False)
    #opt = keras.optimizers.Adam(learning_rate=0.01, epsilon=0.1, decay=0.0)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Train the model for 20 epochs.
    # Shuffle (randomize) order.
    # Update weights after each example (batch_size=1).
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(test_images, test_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        shuffle=True
    )


    # print with tensorboard
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


    # Save the model to disk.
    model.save('handwritten-tf.keras')

if __name__ == '__main__':
    train()














