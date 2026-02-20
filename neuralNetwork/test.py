import os
# Define o nível de log para 3 (apenas erros fatais são mostrados)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("neuralNetwork/models/simple_model.keras")

results = [model.predict(np.array([[60]])),
           model.predict(np.array([[119]])),
           model.predict(np.array([[121]])),
           model.predict(np.array([[87]])),
           model.predict(np.array([[150]]))]

for res in results:
    print("Physical Activity" if res >= 0.5 else "Rest")