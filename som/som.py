from matplotlib.animation import FuncAnimation
import numpy as np
from minisom import MiniSom
import createDataset as cd
import matplotlib.pyplot as plt
import plots as pl


# Split the dataset into training and testing sets
def som(X, num_neurons, num_iterations, learning_rate):
    X_train, y_train, X_test, y_test = cd.splitData(X)


    # Set the map size (dimensions of the grid)
    map_size = (num_neurons, num_neurons)

    # Create the SOM object
    som = MiniSom(*map_size, X_train.shape[1], learning_rate=learning_rate, neighborhood_function='gaussian')

    # Initialize the weights
    som.random_weights_init(X_train)

    # Train the SOM

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    axs[0].set_xlim([-0.2, 1])
    axs[0].set_ylim([-0.2, 1])

    axs[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', label='True Labels (Train)')

    fig.suptitle('Competitive learning', fontsize=16)


    _ = pl.animation(som, X_train, y_train, X_test, y_test, ax=axs[1], fig=fig, num_iteration=num_iterations, active=True, callback=None)

    plt.show()

# n = 504
# X = cd.nonLinearCenter(n)
# som(X, 4, 100, 0.5)
