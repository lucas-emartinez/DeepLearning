import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

def predict_number(image_path):
    # Load the model and metric
    model = keras.models.load_model('./handwritten-tf.keras')


    # Load and preprocess the image (the image is already 28x28)
    image = Image.open(image_path)
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28)


    # Make a prediction
    prediction = model.predict(image)
    return np.argmax(prediction)

# Print the prediction
print(predict_number('./4.jpeg'))
