import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Dataset values
PATH_DATASET = "C:/Users/jstef/Desktop/Projetos_Programação/Python_Vs_Code/hcpa-wearable-monitoring/neuralNetwork/data/dataset2.csv"
TRAIN_RATIO = 0.7
INPUT_QUANTITY = 8
RANDOM_STATE = 42

# Federated Learning Dataset Distribution
ALPHA_DIRICHLET = 0.1

# Neural Network defined values
VALIDATION_SPLIT = 0.2
LOCAL_EPOCHS = 3
BATCH_SIZE = 512
LEARNING_RATE = 0.001

# Federated Learning defined values
NUMBER_HOSPITALS = 5
NUMBER_ROUNDS = 2