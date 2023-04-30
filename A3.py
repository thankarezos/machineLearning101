from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

# Define the 2-variable Gaussian function
s = np.sqrt(2)

def f(x):
    x1, x2 = x
    return np.sqrt(x1**2 + x2**2 + s**2)

# Define the gradient of the 2-variable Gaussian function
def df(x):
    x1, x2 = x
    return np.array([
        x1/np.sqrt(x1**2 + x2**2 + s**2),
        x2/np.sqrt(x1**2 + x2**2 + s**2)
    ])

# Define Gradient Descent
def gradient_descent(f, df, x_init, learning_rate, num_iterations):
    x = np.array(x_init, dtype=np.float64)
    x_history = [x] # to store the history of x values
    for i in range(num_iterations):
        x = x - learning_rate * df(x)
        x_history.append(x)
    return x, np.array(x_history)

def plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    ax.plot_surface(X, Y, Z, cmap='coolwarm', zorder=-1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    scatter = None
    plot = None

    # Define the update function
    def update(frame):
        nonlocal scatter
        nonlocal plot
    
        x_min, x_history = gradient_descent(f, df, x_init, learning_rate, frame+1)

        if plot:
            plot[0].remove()
        plot = ax.plot([x_min[0]], [x_min[1]], [f(x_min)], marker='o', markersize=10, color='black', zorder=5)
        return scatter, plot 

    # Create the animation
    anim = FuncAnimation(fig, update, frames=num_iterations, interval=200)

    # Show the animation
    plt.show()


x_init = np.array([4, -4])

learning_rate = 0.1
num_iterations = 100

plot()


