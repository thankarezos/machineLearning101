from matplotlib.animation import FuncAnimation
import numpy as np
from minisom import MiniSom
import createDataset as cd
import matplotlib.pyplot as plt
import plots as pl


# Split the dataset into training and testing sets
def competitive(X, num_neurons, num_iterations, learning_rate):
    X_train, y_train, X_test, y_test = cd.splitData(X)


    # Set the map size (dimensions of the grid)
    map_size = (num_neurons, 1)

    # Create the SOM object
    som = MiniSom(*map_size, X_train.shape[1], sigma=0.3, learning_rate=learning_rate)

    # Initialize the weights
    som.random_weights_init(X_train)

    # Train the SOM

    fig, ax = plt.subplots(figsize=(12, 8))

    fig.suptitle('Competitive learning', fontsize=16)


    anim = pl.animation(som, X_train, y_train, X_test, y_test, ax=ax, fig=fig, num_iteration=num_iterations, active=True, callback=None)

    plt.show()

