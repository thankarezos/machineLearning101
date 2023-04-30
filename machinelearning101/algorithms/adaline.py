from matplotlib import pyplot as plt
import numpy as np
from . import plots as pl

    

class Adaline:
    def __init__(self, learning_rate=0.001, num_epochs=100):
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
        # return 1 / (1 + np.exp(-X))
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


    def get_x2(self, x1, X_train, X_test):
        if self.weights is None or self.bias is None: 
            raise ValueError("self has not been trained yet.")
        w1, w2 = self.weights
        b = self.bias
        
        x2 = -(w1 * x1 + b - 0.5) / w2

        return x2
    
    def training_finished(self, X_train, y_train, X_test, y_test):
    
        plt.close()
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        fig.suptitle(self.name, fontsize=25)
        fig.text(0.5, 0.92, f"Learning Rate: {self.learning_rate}", ha='center', fontsize=14)
        pl.fit_plot6_static(X_train, y_train, X_test, y_test, axs[0,0])
        pl.fit_plot2_static(self, X_test, X_train, axs[0,1])
        pl.fit_plot5_static(self, X_test, y_test, axs[1,0])
        pl.fit_plot8_static(X_train, y_train, X_test, y_test, axs[1,1])
        plt.show()

    def per_epoch(self, X_train, y_train, X_test, y_test, callback=None):
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

        fig.suptitle(self.name, fontsize=25)
        fig.text(0.5, 0.92, f"Learning Rate: {self.learning_rate}", ha='center', fontsize=14)
        pl.fit_plot6_static(X_train, y_train, X_test, y_test, axs[0,0])
        anim2 = pl.fit_plot2(self, X_train, y_train, X_test, axs[0,1], fig, active=True)
        anim3 = pl.fit_plot3(self, X_train, y_train, X_test, y_test, axs[1,0], fig)
        anim4 = pl.fit_plot7(self, X_train, y_train, X_test, y_test, axs[1,1], fig, callback=callback)
        plt.show()