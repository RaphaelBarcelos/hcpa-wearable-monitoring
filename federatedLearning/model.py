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


def federated_average(weights, sizes):
    new_weights = []

    for layer_weights in zip(*weights):
        new_weights.append(
            np.sum([w * size for w, size in zip(layer_weights, sizes)], axis=0)
            / np.sum(sizes)
        )

    return new_weights