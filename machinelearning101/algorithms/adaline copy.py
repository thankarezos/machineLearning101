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
        self.callback = None

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def activation(self, X):
        return X
    
    def predict(self, X):
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        netinput = self.net_input(X)
        return np.sign(netinput)

    def fit(self, X, y):
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        # Initialize weights and bias to 0
        self.weights = np.random.randn(X.shape[1])
        self.bias = 0

        # Train the ADALINE
        for i in range(self.num_epochs):
            output = self.activation(self.net_input(X))
            errors = (y - output)
            self.weights += self.learning_rate * X.T.dot(errors)
            self.bias += self.learning_rate * errors.sum()
    
    def fit_epoch(self, X, y):
        # Initialize weights and bias to 0 if not already set
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        if self.weights is None:
            self.weights = np.random.randn(X.shape[1])
        if self.bias is None:
            self.bias = 0

        # Train the Adaline for one epoch
        output = self.net_input(X)
        errors = (y - output)
        self.weights += self.learning_rate * X.T.dot(errors)
        self.bias += self.learning_rate * errors.sum()

    def get_x2(self, x1, X_test):
        w1, w2 = self.weights
        b = self.bias
        x2 = -(w1*(x1-np.mean(X_test[:,0]))/np.std(X_test[:,0]) + b) / w2 * np.std(X_test[:,1]) + np.mean(X_test[:,1])
        return x2


    def plot(self, X_test):
        
        y_pred = self.predict(X_test)
        # plt.set_xlim([-0.2, 1])
        # plt.set_ylim([-0.2, 1])

        # Plot the results
        plt.scatter(X_test[:,0], X_test[:,1], c=y_pred)
        x1 = np.linspace(-0.2, 1, 100)
        x2 = self.get_x2(x1, X_test)

        plt.plot(x1, x2)
        plt.show()

    def fit_plot1(self, X_test, y_test, ax, title=None):

        ax.set_xlim([-0.2, 1])
        ax.set_ylim([-0.2, 1])

        ax.set_title(title)
        ax.scatter(X_test[:,0], X_test[:,1], c=y_test)

    def fit_plot2(self, X_train, y_train, X_test, ax, fig, title=None):

        ax.set_xlim([-0.2, 1])
        ax.set_ylim([-0.2, 1])
        
        line, = ax.plot([], [])

        ax.set_title(title)
        title = ax.text(0.1,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")

        # title = ax.set_title(f'Epoch ')
        line, = ax.plot([], [])

        # Define update function
        def update(epoch):
            # Fit the perceptron for one epoch
            self.fit_epoch(X_train, y_train)

            # Predict on X_test
            y_pred = self.predict(X_test)

            scatter = ax.scatter(X_test[:,0], X_test[:,1], c=y_pred)

            # Update line plot
            x1 = np.linspace(-0.2, 1, 100)
            x2 = self.get_x2(x1, X_test)
            line.set_data(x1, x2)

            # Set plot limits and title
            title.set_text(f'Epoch {epoch+1}')

            return scatter, line, title

        anim = FuncAnimation(fig, update, frames=self.num_epochs, blit=True, interval=100, repeat=False)
        return anim
    
    def fit_plot2_static(self, X_test, ax, title=None):

        ax.set_xlim([-0.2, 1])
        ax.set_ylim([-0.2, 1])

        ax.set_title(title)

        y_pred = self.predict(X_test)

        ax.scatter(X_test[:,0], X_test[:,1], c=y_pred)

        # x1 = np.linspace(-0.2, 1, 100)
        # x2 = self.get_x2(x1, X_test)
        # print(x2)

        # ax.plot(x1, x2)

        # Create animation

    def fit_plot3(self, X_train, y_train, X_test, y_test, ax, fig, title=None):
        
        line, = ax.plot([], [])
        # ax.set_ylim([-0.1, 1.1])
        

        ax.set_title(title)
        title = ax.text(0.1, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                        transform=ax.transAxes, ha="center")

        # Define update function
        def update(epoch):
            # Fit the perceptron for one epoch
            self.fit_epoch(X_train, y_train)

            y_pred = self.predict(X_test)

            y_pred_1 = y_pred[y_test == 0]
            y_pred_2 = y_pred[y_test == 1]

            # Create the scatter plot for y_test == 0
            scatter = ax.scatter(range(len(y_pred_1)), y_pred_1, marker='x', c='blue', label='y_test == 0')

            # Create the scatter plot for y_test == 1
            scatter = ax.scatter(range(len(y_pred_1), len(y_test)), y_pred_2, marker='x', c='red', label='y_test == 1')

            # Set plot limits and title
            title.set_text(f'Epoch {epoch + 1}')

            if epoch + 1 == self.num_epochs:
                if self.callback is not None:
                    self.callback()

        # Return plot elements to be updated
            return scatter, line, title
        anim = FuncAnimation(fig, update, frames=self.num_epochs, blit=True, interval=100, repeat=False)
        return anim
    
    def fit_plot3_static(self, X_train, y_train, X_test, y_test, ax, title=None):

        ax.set_title(title)

        self.fit_epoch(X_train, y_train)

        # Predict on X_test
        y_pred = self.predict(X_test)

        y_pred = self.predict(X_test)

        y_pred_1 = y_pred[y_test == 0]
        y_pred_2 = y_pred[y_test == 1]

        # Create the scatter plot for y_test == 0
        scatter = ax.scatter(range(len(y_pred_1)), y_pred_1, marker='x', c='blue', label='y_test == 0')

        # Create the scatter plot for y_test == 1
        scatter = ax.scatter(range(len(y_pred_1), len(y_test)), y_pred_2, marker='x', c='red', label='y_test == 1')


    def fit_plot4(self, X_train, y_train, X_test, y_test, ax, fig, title=None):

        ax.set_xlim([-0.2, 1])
        ax.set_ylim([-0.2, 1])

        line, = ax.plot([], [])

        ax.set_title(title)
        title = ax.text(0.1, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                        transform=ax.transAxes, ha="center")

        # Define update function
        def update(epoch):

            # Fit the perceptron for one epoch
            self.fit_epoch(X_train, y_train)

            # Predict on X_test
            y_pred = self.predict(X_test)

            correct = y_test == y_pred
            scatter = ax.scatter(X_test[correct, 0], X_test[correct, 1], marker='o', color='green')
                    
            incorrect = y_test != y_pred
            
            scatter = ax.scatter(X_test[incorrect, 0], X_test[incorrect, 1], marker='x', color='red')
            # Update line plot
            x1 = np.linspace(-0.2, 1, 100)
            x2 = self.get_x2(x1, X_test)
            line.set_data(x1, x2)

            # Set plot limits and title
            title.set_text(f'Epoch {epoch + 1}')

            if epoch + 1 == self.num_epochs:
                if self.callback is not None:
                    self.callback()

        # Return plot elements to be updated
            return scatter, line, title
        
        anim = FuncAnimation(fig, update, frames=self.num_epochs, blit=True, interval=100, repeat=False )

        return anim
    
    def fit_plot4_static(self, X_train, y_train, X_test, y_test, ax, title=None):

        ax.set_xlim([-0.2, 1])
        ax.set_ylim([-0.2, 1])

        ax.set_title(title)

        self.fit_epoch(X_train, y_train)

        # Predict on X_test
        y_pred = self.predict(X_test)

        correct = y_test == y_pred
        ax.scatter(X_test[correct, 0], X_test[correct, 1], marker='o', color='green')
                
        incorrect = y_test != y_pred
        
        ax.scatter(X_test[incorrect, 0], X_test[incorrect, 1], marker='x', color='red')
        # Update line plot
        x1 = np.linspace(-0.2, 1, 100)
        x2 = self.get_x2(x1, X_test)
        ax.plot(x1, x2)