from matplotlib.animation import FuncAnimation
import numpy as np
from minisom import MiniSom
import createDataset as cd
import matplotlib.pyplot as plt

# Generate linearly separable dataset
n = 504
dataset = cd.nonLinearCenter(n)

# Split the dataset into training and testing sets
X_train, y_train, X_test, y_test = cd.splitData(dataset)

# Set the number of neurons for this dataset

num_neurons = 8

# Set the map size (dimensions of the grid)
map_size = (num_neurons, 1)

# Create the SOM object
som = MiniSom(*map_size, X_train.shape[1], sigma=0.3, learning_rate=0.5)

# Initialize the weights
som.random_weights_init(X_train)

# Train the SOM
num_iterations = 1000
# som.train(X_train, num_iteration=num_iterations)

# Get the winning neurons for each input pattern



# Plot the test data and winning neurons

def animation(model, X_train, y_train, X_test, y_test, ax, fig, num_iteration, title=None, active=False, callback=None):
    ax.set_xlim([-0.2, 1])
    ax.set_ylim([-0.2, 1])
    line, = ax.plot([], [])
    epochs = num_iteration
    def update(epoch):


        for x in X_train:
            model.update(x, som.winner(x), epoch, epochs)
        # rand_idx = np.random.randint(0, X_train.shape[0])
        # model.update(X_train[rand_idx], som.winner(X_train[rand_idx]), epoch, num_iterations)


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

        scatter1 = ax.scatter(X_test[:, 0], X_test[:, 1], c=predicted_labels_test, cmap='coolwarm', label='Predicted Labels (Test)')
        centroids = model.get_weights()
        scatters2 = []
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for i, centroid in enumerate(centroids):
            color = colors[i % len(colors)]  # Get a unique color for each centroid
            scatter = ax.scatter(centroid[:, 0], centroid[:, 1], marker='s', s=50, linewidths=10, color=color)
            scatters2.append(scatter)
        return scatter1, *scatters2,

    anim = FuncAnimation(fig, update, frames=epochs, blit=True, interval=10, repeat=False )
    return anim


fig, ax = plt.subplots(figsize=(12, 8))

fig.suptitle('Competitive learning', fontsize=16)

anim = animation(som, X_train, y_train, X_test, y_test, ax=ax, fig=fig, num_iteration=num_iterations, active=True, callback=None)

plt.show()

