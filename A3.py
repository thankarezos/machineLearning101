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
        -x1/np.sqrt(x1**2 + x2**2 + s**2),
        -x2/np.sqrt(x1**2 + x2**2 + s**2)
    ])

# Define Gradient Descent
def gradient_descent(f, df, x_init, learning_rate, num_iterations):
    x = np.array(x_init, dtype=np.float64)
    x_history = [x] # to store the history of x values
    for i in range(num_iterations):
        x = x - learning_rate * df(x)
        x_history.append(x)
    return x, np.array(x_history)

# Define initial values
x_init = np.array([0,0])

learning_rate = 0.01
num_iterations = 100

x_min, x_history = gradient_descent(f, df, x_init, learning_rate, num_iterations)
print("Minimum point: ", x_min)

x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

# Plot the function and the minimum point
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='coolwarm')
ax.scatter(x_min[0], x_min[1], f(x_min), color='black', s=200,)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# x_history = np.array(x_history)
# ax.plot(x_history[:,0], x_history[:,1], f(x_history.T), '-o', color='black')

plt.show()

# Define the x values for plotting

