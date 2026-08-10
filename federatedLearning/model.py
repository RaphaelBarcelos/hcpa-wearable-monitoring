import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import *

def create_model(numClass):

    model = keras.Sequential([

        layers.Input(shape=(INPUT_QUANTITY,)),

        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation="relu"),

        layers.Dense(numClass, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )

    return model

def fast_predict(model, x_data):

    probs = model(x_data, training=False).numpy()
    return np.argmax(probs, axis=1)

def federated_average(weights, sizes):

    new_weights = []

    for layer_weights in zip(*weights):

        weigth = np.sum([w * size for w, size in zip(layer_weights, sizes)], axis=0) / np.sum(sizes)
        new_weights.append(weigth)

    return new_weights