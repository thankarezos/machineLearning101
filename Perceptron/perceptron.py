import time
from matplotlib import pyplot as plt
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, num_epochs=100):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        # Initialize weights and bias to 0
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Train the perceptron
        for epoch in range(self.num_epochs):
            for i in range(X.shape[0]):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation(linear_output)
                update = self.learning_rate * (y[i] - y_pred)
                self.weights += update * X[i]
                self.bias += update

    def plot(self, X_test):
        y_pred = self.predict(X_test)

        # Plot the results
        plt.scatter(X_test[:,0], X_test[:,1], c=y_pred)
        w1, w2 = self.weights
        b = self.bias
        x1 = np.linspace(-0.2, 1, 100)
        x2 = -(w1*x1 + b) / w2

        plt.xlim(-0.2, 1)
        plt.ylim(-0.2, 1)
        plt.plot(x1, x2)
        plt.show()
    
    def fit_epoch(self, X, y):
        # Initialize weights and bias to 0
        if self.weights is None:
            self.weights = np.zeros(X.shape[1])
        if self.bias is None:
            self.bias = 0

        # Train the perceptron
        for i in range(X.shape[0]):
            linear_output = np.dot(X[i], self.weights) + self.bias
            y_pred = self.activation(linear_output)
            update = self.learning_rate * (y[i] - y_pred)
            self.weights += update * X[i]
            self.bias += update
    
    def fit_plot(self, X_train, y_train, X_test):
        fig, ax = plt.subplots()

        for epoch in range(self.num_epochs):
            # Fit the perceptron for one epoch
            self.fit_epoch(X_train, y_train)

            #predict
            y_pred = self.predict(X_test)

            ax.clear()


            #scatter plot
            ax.scatter(X_test[:,0], X_test[:,1], c=y_pred)

            #plot line
            w1, w2 = self.weights
            b = self.bias
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
    

    def predict(self, X):
        # Compute linear output
        linear_output = np.dot(X, self.weights) + self.bias

        # Apply activation function
        y_pred = np.array([self.activation(x) for x in linear_output])

        return y_pred
