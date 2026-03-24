# Importing Tensor Flow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KERAS_BACKEND"] = "tensorflow"

# Importing Keras
import keras
from keras import layers

# Function to create the Neural Network model
def create_model():

    # Building the neural network structure
    model = keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    # Compiling the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model