import flwr as fl
import pandas as pd
import joblib

from randomForest import RandomForest

import os

# Pega o caminho absoluto da pasta onde o script está (Project/hospitals)
dir_atual = os.path.dirname(os.path.abspath(__file__))

# Sobe um nível e entra em datasets
caminho_csv = os.path.join(dir_atual, '..', 'datasets', 'dataset2.csv')

df = pd.read_csv(caminho_csv, sep=',') 
df = df.iloc[len(df) // 2:]
randomForest = RandomForest(df)

randomForest.preprocessing()
randomForest.encoding()
randomForest.createModel()

class HospitalClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        print(f"[Hospital 2] Treinando modelo...")
        
        randomForest.model.fit(randomForest.X_train, randomForest.y_train)

        randomForest.results()

        joblib.dump(randomForest.model, "rf_hospital_2.joblib")

        print(f"[Hospital 2] modelo Treinado...")

        return [], len(randomForest.X_train), {}
    
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=HospitalClient()
)