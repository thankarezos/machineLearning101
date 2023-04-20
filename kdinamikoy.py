import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from decimal import Decimal

def f(x):
    return np.power(x, 3) - 3*np.power(x, 2)

def df(x):
    return 3*np.power(x, 2) - 6*x

# Define Gradient Descent
def gradient_descent(f, df, x_init, learning_rate, num_iterations):
    x = np.array(x_init, dtype=np.float64)
    for i in range(num_iterations):
        x = x - learning_rate * df(x)
    return x

def plot(x):
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='Function f(x)')
    line_df, = ax.plot([], [], label='Derivative df(x)')
    line_min, = ax.plot([], [], 'ro', label='Minimum')
    line_roots, = ax.plot([], [], 'bo', label='Roots f(x)')
    line_roots_df, = ax.plot([], [], 'co', label='Roots df(x)')
    ax.plot(x, np.zeros_like(x), '--', color='gray')
    ax.plot(np.zeros_like(x), f(x), '--', color='gray')
    ax.set_title("Function and Roots")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    # Define the initialization function
    def init():
        line.set_data([], [])
        line_df.set_data([], [])
        line_min.set_data([], [])
        line_roots.set_data([], [])
        line_roots_df.set_data([], [])
        return line, line_df, line_min, line_roots, line_roots_df

    # Define the update function
    def update(frame):
        x_min = gradient_descent(f, df, x_init, learning_rate, frame+1)
        x_vals = np.linspace(-1, 4, 100)
        y_vals = f(x_vals)
        y_vals_df = df(x_vals)
        line.set_data(x_vals, y_vals)
        line_df.set_data(x_vals, y_vals_df)
        line_min.set_data(x_min, f(x_min))
        line_roots.set_data([0, 3], [f(0), f(3)])
        line_roots_df.set_data([0, 2], [df(0), df(2)])
        if frame == num_iterations-1:
            anim.event_source.stop()
        return line, line_df, line_min, line_roots, line_roots_df

    # Create the animation
    anim = FuncAnimation(fig, update, frames=num_iterations, init_func=init, blit=True)

    # Show the animation
    plt.show()

# Define initial values
x_init = 10
learning_rate = 0.01
num_iterations = 150

# Run Gradient Descent to find the minimum
x_min = gradient_descent(f, df, x_init, learning_rate, num_iterations)

# Define the x values for plotting
x = np.linspace(-1, 4, 100)

# Plot the function and the minimum point

plot(x)
