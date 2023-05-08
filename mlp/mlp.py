from matplotlib.animation import FuncAnimation
from sklearn.exceptions import ConvergenceWarning
import createDataset as cd
import numpy as np
import warnings
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import plots as pl


# X = cd.linearSeparated(504)
X = cd.nonLinearCenter(504)
# X = cd.nonLinearXOR(504)
# X = cd.nonLinear(504)
X_train, y_train, X_test, y_test = cd.splitData(X)

clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, random_state=0, validation_fraction=0.01, n_iter_no_change=100)
# warnings.filterwarnings("ignore", category=ConvergenceWarning)


def training_finished(model, X_train, y_train, X_test, y_test):
    plt.close()
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    pl.plot1(model, X_train, y_train, X_test, y_test, axs[0,0])
    pl.plot2(model, X_train, y_train, X_test, y_test, axs[0,1])
    pl.plot3(model, X_train, y_train, X_test, y_test, axs[1,0])
    pl.plot4(model, X_train, y_train, X_test, y_test, axs[1,1])
    plt.show()

callback = lambda: training_finished(clf, X_train, y_train, X_test, y_test)
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
anim = pl.animation(clf, X_train, y_train, X_test, y_test, axs, fig, callback=callback)
plt.show()


    
