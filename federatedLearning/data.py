import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

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

    class_index = [np.where(y == i)[0] for i in range(numClass)]
    client_index = [[] for _ in range(NUMBER_HOSPITALS)]

    for c in range(numClass):

        index = class_index[c]
        np.random.shuffle(index)

        proportions = np.random.dirichlet(ALPHA_DIRICHLET * np.ones(NUMBER_HOSPITALS))
        
        counts = np.random.multinomial(len(index), proportions)
        
        current = 0
        for client_id, count in enumerate(counts):

            client_index[client_id].extend(index[current : current + count])
            current += count

    client_data = []

    for i in range(NUMBER_HOSPITALS):

        index = np.array(client_index[i])

        if (len(index) == 0):
            client_data.append((np.empty((0, x.shape[1]), dtype='float32'), np.empty((0,), dtype=y.dtype)))

        else:
            client_data.append((x[index], y[index]))

    return client_data


def splitData(clientData):

    np.random.seed(RANDOM_STATE)
    split_data = []

    for x, y in clientData:
        
        n = len(x)

        if (n == 0):
            split_data.append((x, y, x, y))
            continue

        index = np.random.permutation(n)
        split = max(1, int(n * TRAIN_RATIO))

        train_index = index[:split]
        test_index = index[split:]

        scaler = StandardScaler()

        x_train = scaler.fit_transform(x[train_index])
        y_train = y[train_index]

        if (len(test_index) > 0):

            x_test = scaler.transform(x[test_index])
            y_test = y[test_index]

        else:
            x_test = np.empty((0, x.shape[1]), dtype='float32')
            y_test = np.empty((0,), dtype=y.dtype)

        split_data.append((x_train, y_train, x_test, y_test))

    return split_data

def show_confusion_matrix(all_final_y_true, all_final_y_pred):

    confusionMatrix = confusion_matrix(all_final_y_true, all_final_y_pred, normalize='true')

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusionMatrix, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.title('Matriz de Confusão - Modelo Global (Última Rodada)')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Real')
    plt.tight_layout()

    plt.savefig('matriz_confusao.png', dpi=300)
    plt.close()