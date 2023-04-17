import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from perceptron import Perceptron
import time

#import train data
data = pd.read_excel('train.xlsx')
X = data[['X', 'Y']].values
labels = data['labels'].values

# Create a perceptron and fit the data
perceptron = Perceptron(learning_rate=0.1, num_epochs=100)

#import test data
data = pd.read_excel('test.xlsx')
X_test = data[['X', 'Y']].values

# Plot the results

fig, ax = plt.subplots()

for epoch in range(perceptron.num_epochs):
    # Fit the perceptron for one epoch
    perceptron.fit_epoch(X, labels)

    #predict
    y_pred = perceptron.predict(X_test)

    ax.clear()


    #scatter plot
    ax.scatter(X_test[:,0], X_test[:,1], c=y_pred)

    #plot line
    w1, w2 = perceptron.weights
    b = perceptron.bias
    x1 = np.linspace(-0.2, 1, 100)
    x2 = -(w1*x1 + b) / w2
    ax.plot(x1, x2)

    #plot settings
    ax.set_xlim([-0.2, 1])
    ax.set_ylim([-0.2, 1])
    ax.set_title(f'Epoch {epoch+1}')

    # Show the plot
    plt.draw()
    plt.pause(0.1)
    time.sleep(0.1)
plt.show()