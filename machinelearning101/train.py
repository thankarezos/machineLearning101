from matplotlib import pyplot as plt
import numpy as np
from algorithms.perceptron import Perceptron
import createDataset as cd
import matplotlib.gridspec as gridspec


def linearSeperated(n, learning_rate, num_epochs):

    
    def training_finished(perceptron, X_train, y_train, X_test, y_test):
        
        plt.close()
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        
        perceptron.fit_plot1(X_test, y_test, axs[0,0])
        perceptron.fit_plot2_static(X_test, axs[0,1])
        perceptron.fit_plot3_static(X_train, y_train, X_test, y_test, axs[1,0])
        perceptron.fit_plot4_static(X_train, y_train, X_test, y_test, axs[1,1])

        plt.show()

    def per_epoch(perceptron, X_train, y_train, X_test, y_test):
        
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 1], width_ratios=[1, 1])
        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]

        perceptron.fit_plot1(X_test, y_test, axs[0])
        anim2 = perceptron.fit_plot2(X_train, y_train, X_test, axs[1], fig)
        anim3 = perceptron.fit_plot3(X_train, y_train, X_test, y_test, axs[2], fig)

        plt.show()

    X = cd.linearSeparated(n)
    X_train, y_train, X_test, y_test = cd.splitData(X)

    perceptron = [Perceptron(learning_rate=learning_rate, num_epochs=num_epochs)]
    perceptron[0].callback = lambda: training_finished(perceptron[0], X_train, y_train, X_test, y_test)
    per_epoch(perceptron[0], X_train, y_train, X_test, y_test)


linearSeperated(504, 0.01, 100)