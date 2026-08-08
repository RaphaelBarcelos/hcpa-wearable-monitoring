from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import *  
from data import *
from model import *

x, y, numClass = loadData()
partitions = partitionDataByDirichlet(x, y, numClass)
clients_data = splitData(partitions)

global_model = create_model(numClass)
global_weights = global_model.get_weights()

for round_num in range(NUMBER_ROUNDS):

    print(f"\n=== ROUND {round_num+1} ===")

    client_weights = []
    client_sizes = []

    client_metrics = []

    for client_id, (x_train, y_train, x_test, y_test) in enumerate(clients_data):

        if len(x_train) == 0:
            continue

        tf.keras.backend.clear_session()

        local_model = create_model(numClass)
        local_model.set_weights(global_weights)

        local_model.fit(
            x_train,
            y_train,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        client_weights.append(local_model.get_weights())
        client_sizes.append(len(x_train))

        if len(x_test) > 0:
            y_pred = local_model.predict(x_test, verbose=0)
            y_pred = np.argmax(y_pred, axis=1)

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            print(f"[Client {client_id}] Acc: {acc:.4f}")

            client_metrics.append((len(x_test), {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }))

    global_weights = federated_average(client_weights, client_sizes)
    global_model.set_weights(global_weights)

    if len(client_metrics) > 0:

        results = {}
        keys = client_metrics[0][1].keys()
        total = sum(n for n, _ in client_metrics)

        for key in keys:
            mean = sum(n * m[key] for n, m in client_metrics) / total
            results[key] = mean

            var = sum(n * ((m[key] - mean) ** 2) for n, m in client_metrics) / total
            results[f"{key}_var"] = var

        print("\n[GLOBAL METRICS]")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")