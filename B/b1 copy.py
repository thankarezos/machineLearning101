import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
import time

from percetron import Perceptron

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
perceptron = Perceptron(learning_rate=0.00001, num_epochs=100)

# bias = -perceptron.weights[0]/perceptron.weights[1]
bias = -0.4



fig, ax = plt.subplots()



X_0_test = np.random.uniform(low=low1, high=high1, size=(int(n/2), 2))  # Class 0 patterns with x and y in [0.0, 0.3]
X_1_test = np.random.uniform(low=low2, high=high2, size=(int(n/2), 2))  # Class 1 patterns with x and y in [0.7, 0.9]
X_test = np.concatenate((X_0_test, X_1_test), axis=0)

for epoch in range(perceptron.num_epochs):
    # Fit the perceptron for one epoch
    perceptron.fit_epoch(X, labels)
    perceptron.weights[-1] = bias

    y_pred = perceptron.predict(X_test)

    ax.clear()
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
    x1 = np.linspace(0, 1, 100)
    x2 = -(perceptron.weights[0]*x1 + perceptron.weights[-1]) / perceptron.weights[1]
    ax.plot(x1, x2)
    ax.set_xlim([-0.2, 1])
    ax.set_ylim([-0.2, 1])
    ax.set_title(f'Epoch {epoch+1}')
    plt.draw()
    plt.pause(0.01)
    time.sleep(0.01)

# Show the plot
plt.show()

print("finished")