import os
# Define o nível de log para 3 (apenas erros fatais são mostrados)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import tensorflow as tf
import numpy as np
import pandas as pd

df_test = pd.read_csv("neuralNetwork/data/test.csv")

x_test = df_test["Value"].to_numpy()
y_test = df_test["Label"].to_numpy()

print(y_test.shape)

model = tf.keras.models.load_model("neuralNetwork/models/simple_model.keras")

y_pred = model.predict(x_test)
y_pred_classes = tf.argmax(y_pred, axis=1)

y_true = y_test

conf_matrix = tf.math.confusion_matrix(y_true, y_pred_classes)

print(conf_matrix)

#results = [model.predict(np.array([[60]])),
#          model.predict(np.array([[119]])),
#          model.predict(np.array([[121]])),
#          model.predict(np.array([[87]])),
#          model.predict(np.array([[150]]))]

#for res in results:
#    print("Physical Activity" if res >= 0.5 else "Rest")