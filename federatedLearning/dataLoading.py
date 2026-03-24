import numpy as np
import pandas as pd

def loadData():

    df = pd.read_csv("data/dataset2.csv")
    df = df.dropna()
 
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df["activityID"] = df["activityID"].astype("category")
    categoriesLabel = df['activityID'].cat.categories
    df['activityID'] = df['activityID'].cat.codes

    numClass = len(categoriesLabel)

    x = df[["heart_rate",
                    "hand temperature (°C)",
                    "hand acceleration X ±16g",
                    "hand acceleration Y ±16g",
                    "hand acceleration Z ±16g",
                    "hand gyroscope X",
                    "hand gyroscope Y",
                    "hand gyroscope Z"
                    ]].to_numpy().astype('float32')
    y = df["activityID"].to_numpy()

    return x, y, numClass

def partitionDataByDirichlet(x, y, numClass, numClients, alpha):
    
    class_indices = [np.where(y == i)[0] for i in range(numClass)]

    client_indices = [[] for _ in range(numClients)]

    for c in range(numClass):
        indices = class_indices[c]
        np.random.shuffle(indices)

        proportions = np.random.dirichlet(alpha * np.ones(numClients))

        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]

        split_indices = np.split(indices, proportions)

        for client_id, idx in enumerate(split_indices):
            client_indices[client_id].extend(idx)

    client_data = []

    for i in range(numClients):
        idx = client_indices[i]
        client_x = x[idx]
        client_y = y[idx]

        client_data.append((client_x, client_y))

    return client_data

def splitData(clientData, trainRatio):
    
    split_data = []

    for x, y in clientData:
        n = len(x)
        idx = np.random.permutation(n)

        split = int(n * trainRatio)

        train_idx = idx[:split]
        test_idx = idx[split:]

        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        split_data.append((x_train, y_train, x_test, y_test))

    return split_data