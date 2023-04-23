from matplotlib import pyplot as plt
from algorithms.perceptron import Perceptron
from algorithms.adaline import Adaline
import createDataset as cd
import matplotlib.gridspec as gridspec
import plots as pl

def training_finished(model, X_train, y_train, X_test, y_test):
    
    plt.close()
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.suptitle(model.name, fontsize=25)
    fig.text(0.5, 0.92, f"Learning Rate: {model.learning_rate}", ha='center', fontsize=14)
    pl.fit_plot1_static(model, X_test, y_test, axs[0,0])
    pl.fit_plot2_static(model, X_test, axs[0,1])
    pl.fit_plot3_static(model, X_train, y_train, X_test, y_test, axs[1,0])
    pl.fit_plot4_static(model, X_train, y_train, X_test, y_test, axs[1,1])
    plt.show()

def per_epoch(model, X_train, y_train, X_test, y_test, callback=None):
    
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(model.name, fontsize=25)
    fig.text(0.5, 0.92, f"Learning Rate: {model.learning_rate}", ha='center', fontsize=14)
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 1], width_ratios=[1, 1])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    pl.fit_plot1_static(model, X_test, y_test, axs[0])
    anim2 = pl.fit_plot2(model, X_train, y_train, X_test, axs[1], fig, active=True)
    anim3 = pl.fit_plot3(model, X_train, y_train, X_test, y_test, axs[2], fig, callback=callback)
    plt.show()

def linearSeparated(n, model):
    X = cd.linearSeparated(n)
    X_train, y_train, X_test, y_test = cd.splitData(X)
    callback = lambda: training_finished(model,  X_train, y_train, X_test, y_test)
    per_epoch(model, X_train, y_train, X_test, y_test, callback=callback)


# perceptron = Perceptron(learning_rate=0.01, num_epochs=100)
# linearSeparated(504, perceptron)

adaline = Adaline(learning_rate=0.001, num_epochs=100)
linearSeparated(504,adaline)