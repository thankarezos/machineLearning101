import numpy as np
import pandas as pd

low1=0.0
high1=0.3
low2=0.4
high2=0.9
n = 500

def trainData():
    X_0 = np.random.uniform(low=low1, high=high1, size=(int(n/2), 2))  # Class 0 patterns with x and y in [0.0, 0.3]
    X_1 = np.random.uniform(low=low2, high=high2, size=(int(n/2), 2))  # Class 1 patterns with x and y in [0.7, 0.9]
    X = np.concatenate((X_0, X_1), axis=0)
    labels = np.concatenate((np.zeros(len(X_0)), np.ones(len(X_1)))) 

    data = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1], 'labels': labels})

    # Save the DataFrame to an Excel file
    data.to_excel('train.xlsx', index=False)

def testData():
    X_0_test = np.random.uniform(low=0.0, high=0.3, size=(int(n/2), 2))  # Class 0 patterns with x and y in [0.0, 0.3]
    X_1_test = np.random.uniform(low=0.7, high=0.9, size=(int(n/2), 2))  # Class 1 patterns with x and y in [0.7, 0.9]
    X_test = np.concatenate((X_0_test, X_1_test), axis=0)

    data = pd.DataFrame({'X': X_test[:, 0], 'Y': X_test[:, 1]})

    # Save the DataFrame to an Excel file
    data.to_excel('test.xlsx', index=False)

trainData()
testData()