import os
# Define o nível de log para 3 (apenas erros fatais são mostrados)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import tensorflow as tf
from tensorflow import keras
from keras import layers

import numpy as np
import pandas as pd

# Abrindo os datasets
df_train = pd.read_csv("neuralNetwork/data/train.csv")
df_test = pd.read_csv("neuralNetwork/data/test.csv")

# Separando entre treino e teste, valores e rótulos
x_train = df_train["Value"].to_numpy()
y_train = df_train["Label"].to_numpy()

x_test = df_test["Value"].to_numpy()
y_test = df_test["Label"].to_numpy()

# Criando a rede neural com 1 input e 1 output com 32 e 16 neuronios intermediários
model = keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Compilando o modelo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Treianando o modelo
model.fit(x_train, y_train, epochs=7, batch_size=32)

# Testando o modelo
model.evaluate(x_test, y_test)

# Salvando o modelo
model.save("neuralNetwork/models/simple_model.keras")
