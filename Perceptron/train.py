import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from perceptron import Perceptron
import createDataset as cd






def linearSeperated(n, learning_rate, num_epochs):
    X = cd.linearSeparated(n)
    X_train, y_train, X_test, y_test = cd.splitData(X)
    perceptron = Perceptron(learning_rate=0.1, num_epochs=100)
    perceptron.fit_plot(X_train, y_train, X_test)

# Plot the results