import numpy as np
import createDataset as cd
import matplotlib.pyplot as plt

class CompetitiveLearning:
    def __init__(self, num_neurons, learning_rate, num_epochs):
        self.num_neurons = num_neurons
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.random.rand(num_neurons, 2)   # Initialize weights with random values

    def train(self, X_train, y_train):
        for epoch in range(self.num_epochs):
            for x, y in zip(X_train, y_train):
                distances = np.linalg.norm(self.weights - x, axis=1)
                winner_neuron = np.argmin(distances)
                self.weights[winner_neuron] += self.learning_rate * (x - self.weights[winner_neuron])

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            winner_neuron = np.argmin(np.linalg.norm(self.weights - x, axis=1))
            predictions.append(1 if self.weights[winner_neuron][0] > 0.5 else 0)
        return predictions


# Example usage

# Generate random input data
n = 504
dataset = cd.nonLinearCenter(n)

# Split the dataset into training and testing sets
X_train, y_train, X_test, y_test = cd.splitData(dataset)

# Create a CompetitiveLearning object with 16 neurons
cl = CompetitiveLearning(num_neurons=16, learning_rate=0.1, num_epochs=100)

# Train the competitive learning
cl.train(X_train, y_train)

y_pred = cl.predict(X_test)

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
plt.scatter(cl.weights[:, 0], cl.weights[:, 1], c='green', marker='s', label='Neurons')

plt.show()
