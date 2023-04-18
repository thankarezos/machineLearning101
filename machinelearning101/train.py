from algorithms.perceptron import Perceptron
import createDataset as cd

def linearSeperated(n, learning_rate, num_epochs):
    X = cd.linearSeparated(n)
    X_train, y_train, X_test, y_test = cd.splitData(X)
    perceptron = Perceptron(learning_rate=learning_rate, num_epochs=num_epochs)
    perceptron.fit_plot(X_train, y_train, X_test)