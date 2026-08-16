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

    y_probs = model(x_data, training=False).numpy()
    y_pred = np.argmax(y_probs, axis=1)

    return y_pred, y_probs

def federated_average(weights, sizes):

    new_weights = []

    for layer_weights in zip(*weights):

        weigth = np.sum([w * size for w, size in zip(layer_weights, sizes)], axis=0) / np.sum(sizes)
        new_weights.append(weigth)

    return new_weights

@tf.function
def _fedprox_step(model, x_batch, y_batch, global_weights_tf, class_weights_tensor, loss_fn, mu):
    with tf.GradientTape() as tape:
        preds = model(x_batch, training=True)
        
        # Loss base
        sample_losses = loss_fn(y_batch, preds)
        weights_per_sample = tf.gather(class_weights_tensor, tf.cast(y_batch, tf.int32))
        base_loss = tf.reduce_mean(sample_losses * weights_per_sample)
        
        # Termo Proximal do FedProx
        proximal_term = 0.0
        for local_w, global_w in zip(model.weights, global_weights_tf):
            proximal_term += tf.reduce_sum(tf.square(tf.cast(local_w, tf.float32) - global_w))
            
        total_loss = base_loss + (mu / 2.0) * proximal_term

    grads = tape.gradient(total_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss

def train_client_fedprox(model, global_weights, x_train, y_train, epochs, batch_size, mu, class_weights_dict):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    global_weights_tf = [tf.constant(w, dtype=tf.float32) for w in global_weights]
    
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=2048).batch(batch_size)
    
    num_classes = model.output_shape[-1]
    class_weights_vector = np.ones(num_classes, dtype=np.float32)
    for c, w in class_weights_dict.items():
        class_weights_vector[int(c)] = float(w)
    class_weights_tensor = tf.constant(class_weights_vector, dtype=tf.float32)

    for epoch in range(epochs):
        for x_batch, y_batch in dataset:
            _fedprox_step(model, x_batch, y_batch, global_weights_tf, class_weights_tensor, loss_fn, mu)
            
    return model.get_weights()