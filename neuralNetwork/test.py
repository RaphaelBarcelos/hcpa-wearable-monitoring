import os
# Define o nível de log para 3 (apenas erros fatais são mostrados)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import tensorflow as tf
import numpy as np
import pandas as pd

model = tf.keras.models.load_model("neuralNetwork/models/robust_model4.keras")

df = pd.read_csv("neuralNetwork/data/dataset2.csv")
df = df.dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df["activityID"] = df["activityID"].astype("category")
# Guarda os nomes (ex: "transient activities")
categoriesLabel = df['activityID'].cat.categories
# Transforma em 0, 1, 2...
df['activityID'] = df['activityID'].cat.codes

# Guarda o número de classificações
num_classes = len(categoriesLabel)

# Divide o dataset em 70 e 30%
PERCENT = int(len(df) * 0.7)

df_train = df[:PERCENT]
df_test = df[PERCENT:]

features = ["heart_rate",
                   "hand temperature (°C)",
                   "hand acceleration X ±16g",
                   "hand acceleration Y ±16g",
                   "hand acceleration Z ±16g",
                   "hand gyroscope X",
                   "hand gyroscope Y",
                   "hand gyroscope Z"
                   ]

x_test = df_test[features].to_numpy().astype('float32')
y_test = df_test["activityID"].to_numpy()

result = model.evaluate(x_test, y_test)

for nome, valor in zip(model.metrics_names, result):
    print(f"{nome}: {valor}")

# df_test = pd.read_csv("neuralNetwork/data/train.csv")

# x_test = df_test["Value"].to_numpy()
# y_test = df_test["Label"].to_numpy()

# model = tf.keras.models.load_model("neuralNetwork/models/simple_model2.keras")

# y_pred = model.predict(x_test)
# y_pred_classes = tf.argmax(y_pred, axis=1)

# y_true = y_test

# conf_matrix = tf.math.confusion_matrix(y_true, y_pred_classes)

# print(conf_matrix)