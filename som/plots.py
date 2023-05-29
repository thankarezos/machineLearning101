from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt


def animation(model, X_train, y_train, X_test, y_test, ax, fig, num_iteration, title=None, active=False, callback=None):
    ax.set_xlim([-0.2, 1])
    ax.set_ylim([-0.2, 1])
    title = ax.text(0.2, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
            transform=ax.transAxes, ha="center")
    epochs = num_iteration
    def update(epoch):


        for x in X_train:
            model.update(x, model.winner(x), epoch, epochs)
        # rand_idx = np.random.randint(0, X_train.shape[0])
        # model.update(X_train[rand_idx], som.winner(X_train[rand_idx]), epoch, num_iterations)


        winning_neurons_train = np.array([model.winner(x) for x in X_train])
        winning_neurons_test = np.array([model.winner(x) for x in X_test])

        # Assign class labels to the winning neurons in the training data
        predicted_labels_train = np.zeros(X_train.shape[0])
        try:
            for i, (x, neuron) in enumerate(zip(X_train, winning_neurons_train)):
                predicted_labels_train[i] = y_train[np.logical_and(winning_neurons_train[:, 0] == neuron[0], winning_neurons_train[:, 1] == neuron[1])][0]

        # Assign class labels to the winning neurons in the test data
            predicted_labels_test = np.zeros(X_test.shape[0])
            for i, (x, neuron) in enumerate(zip(X_test, winning_neurons_test)):
                predicted_labels_test[i] = y_train[np.logical_and(winning_neurons_train[:, 0] == neuron[0], winning_neurons_train[:, 1] == neuron[1])][0]
        except IndexError:
            pass
        
        scatter1 = ax.scatter(X_test[:, 0], X_test[:, 1], c=predicted_labels_test, cmap='coolwarm', label='Predicted Labels (Test)')
        centroids = model.get_weights()




        scatters2 = []
        # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for i, centroid in enumerate(centroids):
            # color = colors[i % len(colors)]  # Get a unique color for each centroid
            scatter = ax.scatter(centroid[:, 0], centroid[:, 1], marker='s', s=20, linewidths=10, color='red')
            scatters2.append(scatter)

        scatters3 = []   
        for x in X_train:
            i, y = model.winner(x)
            winner = model.get_weights()[i, y]
            scatter = ax.scatter(winner[0], winner[1], marker='s', s=20, linewidths=10, color='green')
            scatters3.append(scatter)
        

        title.set_text(f'Epoch: {epoch+1}/{epochs}')
        return scatter1, *scatters2, title, *scatters3
        # return *scatter1, title,

    anim = FuncAnimation(fig, update, frames=epochs, blit=True, interval=10, repeat=False )
    return anim
