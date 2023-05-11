import numpy as np

class MLP:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.1, max_iter=100, random_state=None):
        self.hidden_layer_sizes = list(hidden_layer_sizes)  # Convert tuple to list
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights = []

    def _initialize_weights(self, n_features, n_classes):
        layer_sizes = [n_features] + self.hidden_layer_sizes + [n_classes]
        for i in range(1, len(layer_sizes)):
            prev_size, curr_size = layer_sizes[i - 1], layer_sizes[i]
            weights = np.random.randn(prev_size, curr_size)
            self.weights.append(weights)

    def _activation_function(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function: " + self.activation)

    def _gradient_descent(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self._initialize_weights(n_features, n_classes)

        for _ in range(self.max_iter):
            # Forward propagation
            activations = [X]
            for W in self.weights:
                net = np.dot(activations[-1], W)
                activation = self._activation_function(net)
                activations.append(activation)

            # Backpropagation
            error = activations[-1] - y
            deltas = [error]
            for i in range(len(self.weights) - 1, 0, -1):
                delta = np.dot(deltas[0], self.weights[i].T) * (activations[i] > 0)
                deltas.insert(0, delta)

            # Update weights
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i])

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self._gradient_descent(X, y)

    def predict(self, X):
        activations = [X]
        for W in self.weights:
            net = np.dot(activations[-1], W)
            activation = self._activation_function(net)
            activations.append(activation)

        return np.argmax(activations[-1], axis=1)
