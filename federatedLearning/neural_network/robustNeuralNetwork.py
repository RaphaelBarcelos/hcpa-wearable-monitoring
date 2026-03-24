# Importing Tensor Flow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KERAS_BACKEND"] = "tensorflow"

# Importing Keras
import keras
from keras import layers

import warnings
warnings.filterwarnings("ignore")

def create_model(numClass):

    model = keras.Sequential([
        layers.Input(shape=(8,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(numClass, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )

    return model