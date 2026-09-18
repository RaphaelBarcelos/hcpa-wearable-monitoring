import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Dataset values
PATH_DATASET = "C:/Users/jstef/Desktop/Projetos_Programação/Python_Vs_Code/hcpa-wearable-monitoring/neuralNetwork/data/dataset2.csv"
TRAIN_RATIO = 0.7
INPUT_QUANTITY = 8
RANDOM_STATE = 42

# Federated Learning Dataset Distribution
ALPHA_DIRICHLET = 0.5

# Neural Network defined values
VALIDATION_SPLIT = 0.2
LOCAL_EPOCHS = 3
BATCH_SIZE = 512
LEARNING_RATE = 0.001

# Federated Learning defined values
NUMBER_HOSPITALS = 3
NUMBER_ROUNDS = 100

# 1 -> FedAvg
# 2 -> FedProx
# 3 -> FedAdam (FedOpt)
AGGREGATION_METHOD = 2

# To FedProx
MU_PROXIMAL = 0.01

# To FedAdam
SERVER_LR = 0.01        
SERVER_BETA_1 = 0.9     
SERVER_BETA_2 = 0.99    
SERVER_TAU = 1e-3