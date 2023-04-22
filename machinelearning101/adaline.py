from matplotlib import pyplot as plt
import numpy as np
from algorithms.adaline import Adaline
import createDataset as cd
import matplotlib.gridspec as gridspec
import algorithms.plots as pl

def training_finished(model, X_train, y_train, X_test, y_test):
    
    plt.close()
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    
    pl.fit_plot1(model, X_test, y_test, axs[0,0])
    pl.fit_plot2_static(model, X_test, axs[0,1])
    pl.fit_plot3_static(model, X_train, y_train, X_test, y_test, axs[1,0])
    pl.fit_plot4_static(model, X_train, y_train, X_test, y_test, axs[1,1])

    plt.show()

def per_epoch(model, X_train, y_train, X_test, y_test):
    
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 1], width_ratios=[1, 1])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]

    pl.fit_plot1(model, X_test, y_test, axs[0])
    anim2 = pl.fit_plot2(model, X_train, y_train, X_test, axs[1], fig)
    anim3 = pl.fit_plot3(model, X_train, y_train, X_test, y_test, axs[2], fig)

    plt.show()


def linearSeperated(n, learning_rate, num_epochs):

    X = cd.linearSeparated(n)
    X_train, y_train, X_test, y_test = cd.splitData(X)

    adaline = [Adaline(learning_rate=learning_rate, num_epochs=num_epochs)]
    adaline[0].callback = lambda: training_finished(adaline[0],  X_train, y_train, X_test, y_test)
    per_epoch(adaline[0], X_train, y_train, X_test, y_test)


linearSeperated(504, 0.01, 100)