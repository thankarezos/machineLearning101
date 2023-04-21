from matplotlib import animation, pyplot as plt
import numpy as np
from algorithms.perceptron import Perceptron
import createDataset as cd

# def fit_plot_combined(X_train, y_train, X_test, pereptron):
#     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
#     ax1.set_xlim([-0.2, 1])
#     ax1.set_ylim([-0.2, 1])
#     ax1.set_title('Model 1')
#     ax2.set_xlim([-0.2, 1])
#     ax2.set_ylim([-0.2, 1])
#     ax2.set_title('Model 2')

#     fig.tight_layout()

#     # Plot first model
#     _, _, anim1, _ = pereptron.fit_plot2(X_train, y_train, X_test, ax=ax1)

#     # Plot second model
#     _, _, anim2, _ = pereptron.fit_plot3(X_train, y_train, X_test, ax=ax2)

#     def update(frame):
#         anim1._draw_frame(frame)
#         anim2._draw_frame(frame)

#     # Combine animations
#     anim = animation.FuncAnimation(fig, update, frames=min(len(anim1._frames), len(anim2._frames)))

#     return fig, ax1, ax2, anim

def linearSeperated(n, learning_rate, num_epochs):
    X = cd.linearSeparated(n)
    X_train, y_train, X_test, y_test = cd.splitData(X)
    perceptron = Perceptron(learning_rate=learning_rate, num_epochs=num_epochs)

    # Call fit_plot2 and fit_plot3
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    anim1 = perceptron.fit_plot2(X_train, y_train, X_test, axs[0,0], fig)
    anim2 = perceptron.fit_plot3(X_train, y_train, X_test, axs[0,1], fig)

    # fig3, ax3 = plt.subplots()
    # fig4, ax4 = plt.subplots()
    # anim3 = perceptron.fit_plot2(X_train, y_train, X_test, ax3, fig3)
    # anim4 = perceptron.fit_plot3(X_train, y_train, X_test, ax4, fig4)
    
    plt.show()



linearSeperated(504, 0.1, 100)