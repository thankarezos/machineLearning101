import numpy as np
import matplotlib.pyplot as plt
from percetron import Perceptron

# Generate some random data
low1=0.0
high1=0.3
low2=0.7
high2=0.9

n = 500
X_0 = np.random.uniform(low=low1, high=high1, size=(int(n/2), 2))
X_1 = np.random.uniform(low=low2, high=high2, size=(int(n/2), 2))
X = np.concatenate((X_0, X_1), axis=0)
labels = np.concatenate((np.zeros(len(X_0)), np.ones(len(X_1))))

# Create a perceptron and fit the data
perceptron = Perceptron(learning_rate=0.1, num_epochs=100)
perceptron.fit(X, labels)

# Make some predictions

# Make some predictions
X_0_test = np.random.uniform(low=low1, high=high1, size=(int(n/2), 2)) # Class 0 patterns with x and y in [0.0, 0.3]
X_1_test = np.random.uniform(low=low2, high=high2, size=(int(n/2), 2))  # Class 1 patterns with x and y in [0.7, 0.9]
X_test = np.concatenate((X_0_test, X_1_test), axis=0)

y_pred = perceptron.predict(X_test)

# Plot the results
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred)
w1, w2 = perceptron.weights
b = perceptron.bias
x1 = np.linspace(-0.2, 1, 100)
x2 = -(w1*x1 + b) / w2

plt.xlim(-0.2, 1)
plt.ylim(-0.2, 1)
plt.plot(x1, x2)
plt.show()