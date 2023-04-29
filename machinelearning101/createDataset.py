import numpy as np
# import pandas as pd


def linearSeparated(n):
    # np.random.seed(0)
    X_0 = np.random.uniform(low=0.0, high=0.3, size=(int(n/2), 2))  # Class 0 patterns with x and y in [0.0, 0.3]
    X_1 = np.random.uniform(low=0.1, high=0.9, size=(int(n/2), 2))  # Class 1 patterns with x and y in [0.7, 0.9]

    X = np.concatenate((X_0, X_1), axis=0)
    labels = np.concatenate((np.zeros(len(X_0)), np.ones(len(X_1))))

    X = np.hstack((X, labels.reshape(-1, 1))) 
    np.random.shuffle(X)  # Shuffle the rows of X
    return X

def nonLinearAngle(n):
    # np.random.seed(0)
    X_0 = np.random.uniform(low=0.0, high=0.3, size=(int(n/2), 2))  # Class 0 patterns with x and y in [0.0, 0.3]
    X_1_1 = np.random.uniform(low=[0.0, 0.4], high=[0.3, 0.9], size=(int(n/4), 2)) # Class 1 patterns with x in [0.0, 0.3] and y in [0.4, 0.9]
    X_1_2 = np.random.uniform(low=[0.4, 0.0], high=[0.9, 0.9], size=(int(n/4), 2)) # Class 1 patterns with x in [0.4, 0.9] and y in [0.0, 0.9]
    X_1 = np.vstack((X_1_1, X_1_2))

    X = np.concatenate((X_0, X_1), axis=0)
    labels = np.concatenate((np.zeros(len(X_0)), np.ones(len(X_1))))

    X = np.hstack((X, labels.reshape(-1, 1))) 
    np.random.shuffle(X)  # Shuffle the rows of X
    return X

def nonLinearCenter(n):
    # np.random.seed(0)
    X_0 = np.random.uniform(low=0.4, high=0.6, size=(int(n/2), 2))  # Class 0 patterns with x and y in [0.0, 0.3]
    X_1_1 = np.random.uniform(low=[0.0, 0.0], high=[0.9, 0.3], size=(int(n/8), 2)) # Class 1 patterns with x in [0.0, 0.3] and y in [0.4, 0.9]
    X_1_2 = np.random.uniform(low=[0.0, 0.7], high=[0.9, 0.9], size=(int(n/8), 2))
    X_1_3 = np.random.uniform(low=[0.0, 0.0], high=[0.3, 0.9], size=(int(n/8), 2))
    X_1_4 = np.random.uniform(low=[0.7, 0.0], high=[0.9, 0.9], size=(int(n/8), 2))
    X_1 = np.vstack((X_1_1, X_1_2, X_1_3, X_1_4))

    X = np.concatenate((X_0, X_1), axis=0)
    labels = np.concatenate((np.zeros(len(X_0)), np.ones(len(X_1))))

    X = np.hstack((X, labels.reshape(-1, 1))) 
    np.random.shuffle(X)  # Shuffle the rows of X
    return X

def nonLinearXOR(n):
    # np.random.seed(0)
    X_0_1 = np.random.uniform(low=0.4, high=0.3, size=(int(n/2), 2))
    X_0_2 = np.random.uniform(low=0.7, high=0.9, size=(int(n/2), 2))
    X_0 = np.vstack((X_0_1, X_0_2))

    X_1_1 = np.random.uniform(low=[0.7, 0.0], high=[0.9, 0.3], size=(int(n/4), 2))
    X_1_2 = np.random.uniform(low=[0.0, 0.7], high=[0.3, 0.9], size=(int(n/4), 2))
    X_1 = np.vstack((X_1_1, X_1_2))

    X = np.concatenate((X_0, X_1), axis=0)
    labels = np.concatenate((np.zeros(len(X_0)), np.ones(len(X_1))))

    X = np.hstack((X, labels.reshape(-1, 1))) 
    np.random.shuffle(X)  # Shuffle the rows of X
    return X

def nonLinear(n):
    # np.random.seed(0)
    X_0 = np.random.uniform(low=0.0, high=0.5, size=(int(n/2), 2))
    X_1 = np.random.uniform(low=0.3, high=0.9, size=(int(n/2), 2))

    X = np.concatenate((X_0, X_1), axis=0)
    labels = np.concatenate((np.zeros(len(X_0)), np.ones(len(X_1))))

    X = np.hstack((X, labels.reshape(-1, 1))) 
    np.random.shuffle(X)  # Shuffle the rows of X
    return X



def splitData(X):
    X_train, X_test = np.split(X, 2)
    return X_train[:, :-1], X_train[:, -1], X_test[:, :-1], X_test[:, -1]


# def saveData(X, filename, labels=True):
#     if labels:
#         data = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1], 'labels': X[:, 2]})
#     else:
#         data = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1]})
#     data.to_excel(filename + '.xlsx', index=False)


