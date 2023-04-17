import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return x**3 - 3*x**2

# Define the derivative of the function
def df(x):
    return 3*x**2 - 6*x

# Define Gradient Descent
def gradient_descent(f, df, x_init, learning_rate, num_iterations):
    x = x_init
    for i in range(num_iterations):
        x = x - learning_rate * df(x)
    return x

def plot(x):
    fig, ax = plt.subplots()
    ax.plot(x, f(x), label='Function f(x)')
    ax.plot(x, df(x), label='Derivative df(x)')
    ax.plot(x, np.zeros_like(x), '--', color='gray')
    ax.plot(np.zeros_like(x), f(x), '--', color='gray')
    ax.plot(x_min, f(x_min), 'ro', label='Minimum')
    ax.plot([0, 3], [f(0), f(3)], 'bo', label='Roots f(x)')
    ax.plot([0, 2], [df(0), df(2)], 'co', label='Roots df(x)')
    x_min_df = gradient_descent(df, lambda x: 6*x-6, 2, 0.1, 50)
    ax.plot([x_min_df], [df(x_min_df)], 'go', markersize=10, label='Minimum of df(x)')
    ax.set_title("Function and Roots")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.show()

# Define initial values
x_init = 4
learning_rate = 0.1
num_iterations = 50

# Run Gradient Descent to find the minimum
x_min = gradient_descent(f, df, x_init, learning_rate, num_iterations)

# Define the x values for plotting
x = np.linspace(-1, 4, 100)

# Plot the function and the minimum point

plot(x)
