import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl

from percetron import Perceptron

# Generate some random data
low1=0.0
high1=0.3
low2=0.4
high2=0.9

# Generate some random data
n = 500
X_0 = np.random.uniform(low=low1, high=high1, size=(int(n/2), 2))  # Class 0 patterns with x and y in [0.0, 0.3]
X_1 = np.random.uniform(low=low2, high=high2, size=(int(n/2), 2))  # Class 1 patterns with x and y in [0.7, 0.9]
X = np.concatenate((X_0, X_1), axis=0)
labels = np.concatenate((np.zeros(len(X_0)), np.ones(len(X_1)))) 

data = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1], 'labels': labels})

# Save the DataFrame to an Excel file
data.to_excel('data.xlsx', index=False)

# Create a perceptron and fit the data
perceptron = Perceptron(learning_rate=0.1, num_epochs=100)
perceptron.fit(X, labels)

# Tune the bias term to improve the decision boundary
# bias = -perceptron.weights[0]/perceptron.weights[1]
bias = -0.03
print(bias)
perceptron.weights[-1] = bias

# Make some predictions
X_0_test = np.random.uniform(low=0.0, high=0.3, size=(int(n/2), 2))  # Class 0 patterns with x and y in [0.0, 0.3]
X_1_test = np.random.uniform(low=0.7, high=0.9, size=(int(n/2), 2))  # Class 1 patterns with x and y in [0.7, 0.9]
X_test = np.concatenate((X_0_test, X_1_test), axis=0)

y_pred = perceptron.predict(X_test)

print(y_pred)

# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)

# x1 = np.linspace(0, 1, 100)
# x2 = -(perceptron.weights[0]*x1 + perceptron.weights[-1]) / perceptron.weights[1]

# plt.xlim(-0.2, 1)
# plt.ylim(-0.2, 1)
# plt.plot(x1, x2)

# # Show the plot
# plt.show()
