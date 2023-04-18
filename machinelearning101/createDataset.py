import numpy as np
import pandas as pd


def linearSeparated(n):
    # np.random.seed(0)
    X_0 = np.random.uniform(low=0.0, high=0.3, size=(int(n/2), 2))  # Class 0 patterns with x and y in [0.0, 0.3]
    X_1 = np.random.uniform(low=0.7, high=0.9, size=(int(n/2), 2))  # Class 1 patterns with x and y in [0.7, 0.9]

    X = np.concatenate((X_0, X_1), axis=0)
    labels = np.concatenate((np.zeros(len(X_0)), np.ones(len(X_1))))

    X = np.hstack((X, labels.reshape(-1, 1))) 
    np.random.shuffle(X)  # Shuffle the rows of X
    return X


def splitData(X):
    X_train, X_test = np.split(X, 2)
    return X_train[:, :-1], X_train[:, -1], X_test[:, :-1], X_test[:, -1]


def saveData(X, filename, labels=True):
    if labels:
        data = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1], 'labels': X[:, 2]})
    else:
        data = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1]})
    data.to_excel(filename + '.xlsx', index=False)


