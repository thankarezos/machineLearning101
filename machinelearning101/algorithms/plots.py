from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
count = 0

def plot(model, X_test):
    
    y_pred = model.predict(X_test)
    plt.xlim(-0.2, 1)
    plt.ylim(-0.2, 1)

    # Plot the results
    plt.scatter(X_test[:,0], X_test[:,1], c=y_pred)
    x1 = np.linspace(0, 1, 100)
    x2 = model.get_x2(x1, X_test)


    x1 = np.linspace(-0.2, 1, 100)
    x2 = model.get_x2(x1, X_test)
    plt.plot(x1, x2)

    plt.show()

def fit_plot1(model, X_test, y_test, ax, title=None):

    ax.set_xlim([-0.2, 1])
    ax.set_ylim([-0.2, 1])

    ax.set_title(title)
    ax.scatter(X_test[:,0], X_test[:,1], c=y_test)

def fit_plot2(model, X_train, y_train, X_test, ax, fig, title=None, active=False):

    ax.set_xlim([-0.2, 1])
    ax.set_ylim([-0.2, 1])
    
    line, = ax.plot([], [])

    ax.set_title(title)
    title = ax.text(0.1,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
            transform=ax.transAxes, ha="center")

    # title = ax.set_title(f'Epoch ')
    line, = ax.plot([], [])

    # Define update function
    def update(epoch):
        # Fit the perceptron for one epoch
        if model.trained <= epoch and active:
            model.fit_epoch(X_train, y_train)

        # Predict on X_test
        y_pred = model.predict(X_test)

        scatter = ax.scatter(X_test[:,0], X_test[:,1], c=y_pred)

        # Update line plot
        x1 = np.linspace(-0.2, 1, 100)
        x2 = model.get_x2(x1, X_test)
        line.set_data(x1, x2)

        # Set plot limits and title
        title.set_text(f'Epoch {model.trained + 1}')

        return scatter, line, title

    anim = FuncAnimation(fig, update, frames=model.num_epochs, blit=True, interval=100, repeat=False)
    return anim

def fit_plot2_static(model, X_test, ax, title=None):

    ax.set_xlim([-0.2, 1])
    ax.set_ylim([-0.2, 1])

    ax.set_title(title)

    y_pred = model.predict(X_test)

    ax.scatter(X_test[:,0], X_test[:,1], c=y_pred)

    x1 = np.linspace(-0.2, 1, 100)
    x2 = model.get_x2(x1, X_test)

    ax.plot(x1, x2)

def fit_plot3(model, X_train, y_train, X_test, y_test, ax, fig, title=None, active=False):
    
    line, = ax.plot([], [])

    ax.set_title(title)
    title = ax.text(0.1, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                    transform=ax.transAxes, ha="center")
    

    # Define update function
    def update(epoch):

        if model.trained <= epoch and active:
            model.fit_epoch(X_train, y_train)

        y_pred = model.predict(X_test)

        y_pred_1 = y_pred[y_test == 0]
        y_pred_2 = y_pred[y_test == 1]

        # Create the scatter plot for y_test == 0
        scatter = ax.scatter(range(len(y_pred_1)), y_pred_1, marker='x', c='blue', label='y_test == 0')

        # Create the scatter plot for y_test == 1
        scatter = ax.scatter(range(len(y_pred_1), len(y_test)), y_pred_2, marker='x', c='red', label='y_test == 1')

        # Set plot limits and title
        title.set_text(f'Epoch {model.trained + 1}')

        if epoch + 1 == model.num_epochs:
            if model.callback is not None:
                model.callback()

    # Return plot elements to be updated
        return scatter, line, title
    anim = FuncAnimation(fig, update, frames=model.num_epochs, blit=True, interval=100, repeat=False)
    return anim

def fit_plot3_static(model, X_train, y_train, X_test, y_test, ax, title=None):

    ax.set_title(title)

    # Predict on X_test
    y_pred = model.predict(X_test)

    y_pred = model.predict(X_test)

    y_pred_1 = y_pred[y_test == 0]
    y_pred_2 = y_pred[y_test == 1]

    # Create the scatter plot for y_test == 0
    scatter = ax.scatter(range(len(y_pred_1)), y_pred_1, marker='x', c='blue', label='y_test == 0')

    # Create the scatter plot for y_test == 1
    scatter = ax.scatter(range(len(y_pred_1), len(y_test)), y_pred_2, marker='x', c='red', label='y_test == 1')


def fit_plot4(model, X_train, y_train, X_test, y_test, ax, fig, title=None):

    ax.set_xlim([-0.2, 1])
    ax.set_ylim([-0.2, 1])

    line, = ax.plot([], [])

    ax.set_title(title)
    title = ax.text(0.1, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                    transform=ax.transAxes, ha="center")

    # Define update function
    def update(epoch):

        # Fit the perceptron for one epoch

        # Predict on X_test
        y_pred = model.predict(X_test)

        correct = y_test == y_pred
        scatter = ax.scatter(X_test[correct, 0], X_test[correct, 1], marker='o', color='green')
                
        incorrect = y_test != y_pred
        
        scatter = ax.scatter(X_test[incorrect, 0], X_test[incorrect, 1], marker='x', color='red')
        # Update line plot
        x1 = np.linspace(-0.2, 1, 100)
        x2 = model.get_x2(x1, X_test)
        line.set_data(x1, x2)

        # Set plot limits and title
        title.set_text(f'Epoch {model.trained + 1}')

        

        if epoch + 1 == model.num_epochs:
            if model.callback is not None:
                model.callback()

    # Return plot elements to be updated
        return scatter, line, title
    
    anim = FuncAnimation(fig, update, frames=model.num_epochs, blit=True, interval=100, repeat=False )

    return anim

def fit_plot4_static(model, X_train, y_train, X_test, y_test, ax, title=None):

    ax.set_xlim([-0.2, 1])
    ax.set_ylim([-0.2, 1])

    ax.set_title(title)

    # Predict on X_test
    y_pred = model.predict(X_test)

    correct = y_test == y_pred
    ax.scatter(X_test[correct, 0], X_test[correct, 1], marker='o', color='green')
            
    incorrect = y_test != y_pred
    
    ax.scatter(X_test[incorrect, 0], X_test[incorrect, 1], marker='x', color='red')
    # Update line plot
    x1 = np.linspace(-0.2, 1, 100)
    x2 = model.get_x2(x1, X_test)
    ax.plot(x1, x2)