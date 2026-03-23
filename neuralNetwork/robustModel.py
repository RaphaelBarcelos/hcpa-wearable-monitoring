import os
# Define o nível de log para 3 (apenas erros fatais são mostrados)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import tensorflow as tf
from tensorflow import keras
from keras import layers

import numpy as np
import pandas as pd

df = pd.read_csv("neuralNetwork/data/dataset2.csv")
df = df.dropna()

# Deixa o dataset aleatorio 
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Transforma os textos da coluna 'activityID' em categorias numéricas
df["activityID"] = df["activityID"].astype("category")
# Guarda os nomes (ex: "transient activities")
categoriesLabel = df['activityID'].cat.categories
# Transforma em 0, 1, 2...
df['activityID'] = df['activityID'].cat.codes

# Guarda o número de classificações
num_classes = len(categoriesLabel)

# Divide o dataset em 70 e 30%
PERCENT = int(len(df) * 0.7)

df_train = df[PERCENT:]
df_test = df[:PERCENT]

# Separando entre treino e teste, valores e rótulos

features = ["heart_rate",
                   "hand temperature (°C)",
                   "hand acceleration X ±16g",
                   "hand acceleration Y ±16g",
                   "hand acceleration Z ±16g",
                   "hand gyroscope X",
                   "hand gyroscope Y",
                   "hand gyroscope Z"
                   ]

x_train = df_train[features].to_numpy().astype('float32')
y_train = df_train["activityID"].to_numpy()

x_test = df_test[features].to_numpy().astype('float32')
y_test = df_test["activityID"].to_numpy()

print(np.isnan(x_train).sum())
print(np.isinf(x_train).sum())

# Criando a rede neural com 1 input e 1 output com 64 e 32 neuronios intermediários
model = keras.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

# Compilando o modelo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"]
)

# Treinando o modelo
model.fit(x_train, y_train, epochs=7, batch_size=32)

# Testando o modelo
model.evaluate(x_test, y_test)

# Salvando o modelo
model.save("neuralNetwork/models/robust_model2.keras")
