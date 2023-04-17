import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from percetron import Perceptron

#import train data
data = pd.read_excel('train.xlsx')
X = data[['X', 'Y']].values
labels = data['labels'].values

# Create a perceptron and fit the data
perceptron = Perceptron(learning_rate=0.1, num_epochs=100)
perceptron.fit(X, labels)


#import test data
data = pd.read_excel('test.xlsx')
X_test = data[['X', 'Y']].values

# Make some predictions
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