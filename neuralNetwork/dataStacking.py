import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import glob

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

# Coletando todos os datasets
all_archives = glob.glob("datasetScript/finalData/*_heartrate.txt")

df_list_train = []
df_list_test = []

counter = 0

# Passando pelos datasets e adicionando na lista de treino e teste
for path in all_archives:
    df = pd.read_csv(path, header=None, names=["Time", "Value", "Label"])

    df["Label"] = df["Label"].replace({"Rest" : 0, "Physical Activity" : 1})
    df["Value"] = df["Value"].astype(int)

    # Cálculo para separar em 70% para treino e 30% para teste
    percent = len(all_archives) * 0.7

    if counter >= percent:
        df_list_test.append(df)
    else:
        df_list_train.append(df)

    counter+=1

# Empilhando os datasets das listas em um só
df_train = pd.concat(df_list_train)
df_test = pd.concat(df_list_test)

# Salvando os novos datasets
df_train.to_csv("neuralNetwork/data/train.csv", header=["Time", "Value", "Label"], index=False)
df_test.to_csv("neuralNetwork/data/test.csv", header=["Time", "Value", "Label"], index=False)