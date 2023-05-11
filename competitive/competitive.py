import numpy as np
from minisom import MiniSom
import createDataset as cd
import matplotlib.pyplot as plt

# Generate linearly separable dataset
n = 504
dataset = cd.nonLinearXOR(n)

# Split the dataset into training and testing sets
X_train, y_train, X_test, y_test = cd.splitData(dataset)

# Set the number of neurons for this dataset
num_neurons = 4

# Set the map size (dimensions of the grid)
map_size = (num_neurons, 1)

# Create the SOM object
som = MiniSom(*map_size, X_train.shape[1], sigma=0.3, learning_rate=0.5)

# Initialize the weights
som.random_weights_init(X_train)

# Train the SOM
num_iterations = 100
som.train(X_train, num_iteration=num_iterations)

# Get the winning neurons for each input pattern
winning_neurons_train = np.array([som.winner(x) for x in X_train])
winning_neurons_test = np.array([som.winner(x) for x in X_test])

# Assign class labels to the winning neurons in the training data
predicted_labels_train = np.zeros(X_train.shape[0])
for i, (x, neuron) in enumerate(zip(X_train, winning_neurons_train)):
    predicted_labels_train[i] = y_train[np.logical_and(winning_neurons_train[:, 0] == neuron[0], winning_neurons_train[:, 1] == neuron[1])][0]

# Assign class labels to the winning neurons in the test data
predicted_labels_test = np.zeros(X_test.shape[0])
for i, (x, neuron) in enumerate(zip(X_test, winning_neurons_test)):
    predicted_labels_test[i] = y_train[np.logical_and(winning_neurons_train[:, 0] == neuron[0], winning_neurons_train[:, 1] == neuron[1])][0]


# Plot the test data and winning neurons
plt.scatter(X_test[:, 0], X_test[:, 1], c=predicted_labels_test, cmap='coolwarm', label='Predicted Labels (Test)')
plt.scatter(winning_neurons_train[:, 0], winning_neurons_train[:, 1], c='green', marker='s', s=100, label='Winning Neurons (Train)')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Training and Test Data with Winning Neurons')
plt.show()
