from matplotlib import pyplot as plt
from algorithms.perceptron import Perceptron
from algorithms.adaline import Adaline
import createDataset as cd

def train(model, X):
    X_train, y_train, X_test, y_test = cd.splitData(X)
    callback = lambda: model.training_finished(X_train, y_train, X_test, y_test)
    model.per_epoch(X_train, y_train, X_test, y_test, callback)

def linearSeparated(n, model):
    X = cd.linearSeparated(n)
    train(model, X)

def nonlinearSeparatedAngle(n, model):
    X = cd.nonLinearAngle(n)
    train(model, X)

def nonlinearSeparatedCenter(n, model):
    X = cd.nonLinearCenter(n)
    train(model, X)

def nonlinearSeparated(n, model):
    X = cd.nonLinear(n)
    train(model, X)

model = Perceptron(learning_rate=0.01, num_epochs=10)
# model = Adaline(learning_rate=0.001, num_epochs=100)
# linearSeparated(504, model)
# nonlinearSeparatedAngle(504, model)
# nonlinearSeparatedCenter(504, model)
nonlinearSeparated(504, model)