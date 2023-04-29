from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.gridspec as gridspec
from . import plots as pl

class Perceptron:
    def __init__(self, learning_rate=0.1, num_epochs=100):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.bias = None
        self.name = "Perceptron"
        self.trained = 0
        self.weights = None

    def activation(self, x):
        return np.where(x >= 0, 1, 0)
    

    def predict(self, X):
        # Compute linear output
        linear_output = np.dot(X, self.weights) + self.bias
        # Apply activation function
        y_pred = np.array([self.activation(x) for x in linear_output])
        
        return y_pred
    
    def fit(self, X, y):

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
        self.trained += 1
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
            
    def get_x2(self, x1, X_test, X_train):
        w1, w2 = self.weights
        b = self.bias
        x2 = -(w1*x1 + b) / w2
        return x2
    
    def training_finished(self, X_train, y_train, X_test, y_test):
    
        plt.close()
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        fig.suptitle(self.name, fontsize=25)
        fig.text(0.5, 0.92, f"Learning Rate: {self.learning_rate}", ha='center', fontsize=14)
        pl.fit_plot1_static(X_test, y_test, axs[0,0])
        pl.fit_plot2_static(self, X_test, X_train, axs[0,1])
        pl.fit_plot3_static(self, X_test, y_test, axs[1,0])
        pl.fit_plot5_static(self, X_test, y_test, axs[1,1])
        plt.show()

    def per_epoch(self, X_train, y_train, X_test, y_test, callback=None):
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(self.name, fontsize=25)
        fig.text(0.5, 0.92, f"Learning Rate: {self.learning_rate}", ha='center', fontsize=14)
        gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 1], width_ratios=[1, 1])
        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
        pl.fit_plot1_static(X_test, y_test, axs[0])
        anim2 = pl.fit_plot2(self, X_train, y_train, X_test, axs[1], fig, active=True)
        anim3 = pl.fit_plot3(self, X_train, y_train, X_test, y_test, axs[2], fig, callback=callback)
        plt.show()