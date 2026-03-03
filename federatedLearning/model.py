# Importing Tensor Flow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KERAS_BACKEND"] = "tensorflow"

# Importing Keras
import keras
from keras import layers

# Importing Pandas
import pandas as pd

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

# Function to load the data from CSV files
def load_data():

    # Loading the training and testing data from CSV files
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")

    # Transforming the training data into numpy arrays
    x_train = df_train["Value"].to_numpy()
    y_train = df_train["Label"].to_numpy()

    # Transforming the testing data into numpy arrays
    x_test = df_test["Value"].to_numpy()
    y_test = df_test["Label"].to_numpy()

    return (x_train, y_train), (x_test, y_test)

# Function to partition the data into pieces for each client
def partition_data(x, y, num_clients):

    # Defining the total and piece's size
    data_size = len(x)
    piece_size = data_size // num_clients
    partitions = []

    # Partitioning the data into pieces for each client
    for i in range(num_clients):

        start = i * piece_size
        end = 0

        if i != num_clients - 1:
            end = (i + 1) * piece_size
        else:
            end = data_size

        partitions.append((x[start:end], y[start:end]))

    return partitions