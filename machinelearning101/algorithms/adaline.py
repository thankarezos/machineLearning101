from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

    

class Adaline:
    def __init__(self, learning_rate=0.05, num_epochs=100, callback=None):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.lr = learning_rate
        self.bias = None
        self.name = "Adaline"
        self.trained = 0
    
    def net_input(self, X):
        weighted_sum = np.dot(X, self.weights) + self.bias
        return weighted_sum
    
    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

    

    def fit(self, X, y):
        self.weights = np.random.rand(X.shape[1])
        self.bias = 0.0
        for epoch in range(self.num_epochs):
            for xi, target in zip(X, y):
                net_input = self.net_input(xi)
                output = self.activation(net_input)
                error = (target - output)
                self.weights += self.learning_rate * xi * error
                self.bias += self.learning_rate * error
                cost = 0.5 * error ** 2
            return cost

    
    def fit_epoch(self, X, y):
        self.trained += 1

        if self.weights is None:
            self.weights = np.random.rand(X.shape[1])
        if self.bias is None:
            self.bias = 0.0
        cost = 0.0
        for xi, target in zip(X, y):
                net_input = self.net_input(xi)
                output = self.activation(net_input)
                error = (target - output)
                self.weights += self.learning_rate * xi * error
                self.bias += self.learning_rate * error
                cost = 0.5 * error ** 2
        return cost

            

    def get_x2(self, x1, X_test):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet.")
        w1, w2 = self.weights
        b = self.bias
        
        # standardize x1 using mean and std from training data
        x1_std = (x1 - np.mean(X_test[:, 0])) / np.std(X_test[:, 0])
        
        # compute corresponding x2 value
        x2 = -(w1 * x1_std + b) / w2

        # unstandardize x2 using mean and std from training data
        x2 = x2 * np.std(X_test[:, 0]) + np.mean(X_test[:, 0])

        return x2