import numpy as np
import matplotlib.pyplot as plt

# Define the 2-variable Gaussian function
def f(x):
    x1, x2 = x
    return np.exp(-(x1**2 + x2**2)/2)

# Define the gradient of the 2-variable Gaussian function
def df(x):
    x1, x2 = x
    return np.array([
        -x1*np.exp(-(x1**2 + x2**2)/2),
        -x2*np.exp(-(x1**2 + x2**2)/2)
    ])

# Define Gradient Descent
def gradient_descent(f, df, x_init, learning_rate, num_iterations):
    x = x_init
    for i in range(num_iterations):
        x = x - learning_rate * df(x)
    return x

# Define initial values
x_init = np.array([0, 0])
learning_rate = 0.1
num_iterations = 50

# Run Gradient Descent to find the minimum
x_min = gradient_descent(f, df, x_init, learning_rate, num_iterations)

# Print the minimum point
print("Minimum point: ", x_min)

# Define the x and y values for plotting
x = np.linspace(-3, 5, 100)
y = np.linspace(-5, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

# Plot the function and the minimum point
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='coolwarm')
ax.scatter(x_min[0], x_min[1], f(x_min), color='red')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
