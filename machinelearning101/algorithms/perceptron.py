from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, num_epochs=100, callback=None):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None
        self.callback = callback
        self.name = "Perceptron"

    def activation(self, x):
        return np.where(x >= 0, 1, 0)
    

    def predict(self, X):
        # Compute linear output
        linear_output = np.dot(X, self.weights) + self.bias
        # Apply activation function
        y_pred = np.array([self.activation(x) for x in linear_output])
        
        return y_pred
    
    def fit(self, X, y):
        # Initialize weights and bias to 0
        self.weights = np.random.rand(X.shape[1])
        self.bias = 0

        # Train the perceptron
        for epoch in range(self.num_epochs):
            for i in range(X.shape[0]):
                linear_output = np.dot(X[i], self.weights) + self.bias

                y_pred = self.activation(linear_output)
                
                update = self.learning_rate * (y[i] - y_pred)
                self.weights += update * X[i]
                self.bias += update
    
    def fit_epoch(self, X, y):
        # Initialize weights and bias to 0
        if self.weights is None:
            self.weights = np.random.rand(X.shape[1])
        if self.bias is None:
            self.bias = 0

        # Train the perceptron
        for i in range(X.shape[0]):
            linear_output = np.dot(X[i], self.weights) + self.bias
            y_pred = self.activation(linear_output)
            update = self.learning_rate * (y[i] - y_pred)
            self.weights += update * X[i]
            self.bias += update
            
    def get_x2(self, x1, X_test):
        w1, w2 = self.weights
        b = self.bias
        x2 = -(w1*x1 + b) / w2
        return x2