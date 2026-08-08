import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import *  

def loadData():

    df = pd.read_csv(PATH_DATASET)
    df = df.dropna()

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    df["activityID"] = df["activityID"].astype("category")
    categoriesLabel = df['activityID'].cat.categories
    df['activityID'] = df['activityID'].cat.codes

    numClass = len(categoriesLabel)

    x = df[[
        "heart_rate",
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


def partitionDataByDirichlet(x, y, numClass):

    np.random.seed(RANDOM_STATE)

    class_indices = [np.where(y == i)[0] for i in range(numClass)]
    client_indices = [[] for _ in range(NUMBER_HOSPITALS)]

    for c in range(numClass):
        indices = class_indices[c]
        np.random.shuffle(indices)

        proportions = np.random.dirichlet(ALPHA_DIRICHLET * np.ones(NUMBER_HOSPITALS))
        proportions = np.maximum(proportions, 1e-6)
        proportions = proportions / proportions.sum()

        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)

        for client_id, idx in enumerate(split_indices):
            client_indices[client_id].extend(idx)

    client_data = []

    for i in range(NUMBER_HOSPITALS):
        idx = client_indices[i]

        if len(idx) == 0:
            client_data.append((np.array([]), np.array([])))
        else:
            client_data.append((x[idx], y[idx]))

    return client_data


def splitData(clientData):

    np.random.seed(RANDOM_STATE)

    split_data = []

    for x, y in clientData:
        n = len(x)

        if n == 0:
            split_data.append((x, y, x, y))
            continue

        idx = np.random.permutation(n)

        split = max(1, int(n * TRAIN_RATIO))

        train_idx = idx[:split]
        test_idx = idx[split:]

        scaler = StandardScaler()

        x_train, y_train = scaler.fit_transform(x[train_idx]), y[train_idx]
        x_test, y_test = scaler.transform(x[test_idx]), y[test_idx]

        split_data.append((x_train, y_train, x_test, y_test))

    return split_data