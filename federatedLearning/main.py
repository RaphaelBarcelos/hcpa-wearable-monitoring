import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataLoading import loadData, partitionDataByDirichlet, splitData
from neural_network.robustNeuralNetwork import create_model

import logging
logging.getLogger("flwr").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

num_hospitals = int(input("Enter the number of hospitals: "))
alpha = 0.1

X, y, numClass = loadData()

partitions = partitionDataByDirichlet(X, y, numClass, num_hospitals, alpha)

clients_data = splitData(partitions, trainRatio=0.8)

global_model = create_model(numClass)

def client(cid: str):

    client_id = int(cid)

    x_train, y_train, x_test, y_test = clients_data[client_id]

    if len(x_train) == 0:

        class EmptyClient(fl.client.NumPyClient):
            def get_parameters(self, config):
                return model.get_weights()

            def fit(self, parameters, config):
                return parameters, 1, {}

            def evaluate(self, parameters, config):

                print(f"[Client {client_id}] "
                f"Loss: {0:.4f} | Acc: {0:.4f} | "
                f"Prec: {0:.4f} | Rec: {0:.4f} | F1: {0:.4f}")

                return 0.0, 1, {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                }

        return EmptyClient()

    tf.keras.backend.clear_session()
    model = create_model(numClass)

    class HospitalClient(fl.client.NumPyClient):

        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):

            model.set_weights(parameters)

            model.fit(
                x_train,
                y_train,
                epochs=7,
                batch_size=32,
                verbose=0
            )

            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):

            model.set_weights(parameters)

            if len(x_test) == 0:

                print(f"[Client {client_id}] "
                f"Loss: {0:.4f} | Acc: {0:.4f} | "
                f"Prec: {0:.4f} | Rec: {0:.4f} | F1: {0:.4f}")

                return 0.0, 0, {"accuracy": 0.0}

            loss = model.evaluate(x_test, y_test, verbose=0)[0]

            y_pred = model.predict(x_test, verbose=0)
            y_pred = np.argmax(y_pred, axis=1)

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            print(f"[Client {client_id}] "
                f"Loss: {loss:.4f} | Acc: {acc:.4f} | "
                f"Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")

            return loss, len(x_test), {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

    return HospitalClient()

def weighted_average(metrics):

    results = {}
    keys = metrics[0][1].keys()

    total_examples = sum(num_examples for num_examples, _ in metrics)

    for key in keys:
        values = [num_examples * m[key] for num_examples, m in metrics]
        mean = sum(values) / total_examples

        results[key] = mean

        var_values = [
            num_examples * ((m[key] - mean) ** 2)
            for num_examples, m in metrics
        ]

        variance = sum(var_values) / total_examples

        results[f"{key}_var"] = variance
    
    print("\n[GLOBAL METRICS]")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print()

    return results

strategy = fl.server.strategy.FedAvg(
    min_available_clients=num_hospitals,
    evaluate_metrics_aggregation_fn=weighted_average
)

history = fl.simulation.start_simulation(
    client_fn=client,
    num_clients=num_hospitals,
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=strategy,
    ray_init_args={"local_mode": True}
)

final_parameters = strategy.parameters
global_model.set_weights(fl.common.parameters_to_ndarrays(final_parameters))
global_model.save("models/federated_model.keras")