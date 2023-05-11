from matplotlib.animation import FuncAnimation
from sklearn.exceptions import ConvergenceWarning
import createDataset as cd
import numpy as np
import warnings
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import plots as pl
from gradientMLP import MLP as mlpGD

# X = cd.linearSeparated(504)
# X = cd.nonLinearCenter(504)
# # X = cd.nonLinearXOR(504)
# # X = cd.nonLinear(504)
# X_train, y_train, X_test, y_test = cd.splitData(X)

# clf = mlpGD(hidden_layer_sizes=(100,), activation='relu', max_iter=100, random_state=0)
# clf.fit(X_train, y_train)
# clf.predict(X_test)

def train(model, title, X_train, y_train, X_test, y_test):
    def training_finished(model, X_train, y_train, X_test, y_test):
        plt.close()
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
        fig.suptitle(title)
        pl.plot1(model, X_train, y_train, X_test, y_test, axs[0,0])
        pl.plot2(model, X_train, y_train, X_test, y_test, axs[0,1])
        pl.plot3(model, X_train, y_train, X_test, y_test, axs[1,0])
        pl.plot4(model, X_train, y_train, X_test, y_test, axs[1,1])
        plt.show()

    callback = lambda: training_finished(model, X_train, y_train, X_test, y_test)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    fig.suptitle(title)
    anim = pl.animation(model, X_train, y_train, X_test, y_test, axs, fig, callback=callback)
    plt.show()

def SGD(X):
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='sgd', max_iter=1000, random_state=0, validation_fraction=0.01, n_iter_no_change=100, learning_rate="adaptive", learning_rate_init=0.1)
    title = "SGD"
    X_train, y_train, X_test, y_test = cd.splitData(X)
    train(clf, title, X_train, y_train, X_test, y_test)

    
def GD(X):
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='sgd', max_iter=1000, random_state=0, validation_fraction=0.01, n_iter_no_change=100, learning_rate='constant', learning_rate_init=0.1)
    title = "GD"
    X_train, y_train, X_test, y_test = cd.splitData(X)
    train(clf, title, X_train, y_train, X_test, y_test)
    


# GD()
SGD()




# warnings.filterwarnings("ignore", category=ConvergenceWarning)





    
