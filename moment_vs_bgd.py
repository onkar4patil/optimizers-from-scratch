# moment gradient descent vs batch gradient descent
import numpy as np
import matplotlib.pyplot as plt

# %%
# quadratic loss function
def quadratic_loss(x, y):
    return x**2 + 10*y**2

# %%
#gradient of the loss function
def quadratic_grad(x, y):
    dx = 2 * x
    dy = 10 * 2 * y
    return np.array([dx, dy])

# %%
#batch GD
def batch_gradient_descent(grad_func, eta, epochs, start_point):
    x, y = start_point
    path = [(x, y)]
    losses = [quadratic_loss(x, y)]

    for _ in range(epochs):
        grad = grad_func(x, y)
        x -= eta * grad[0]
        y -= eta * grad[1]
        path.append((x, y))
        losses.append(quadratic_loss(x, y))

    return np.array(path), losses

# %%
#gradient descent with momentum
def gradient_descent_momentum(grad_func, eta, beta, epochs, start_point):
    x, y = start_point
    v = np.array([0,0])
    path = [(x, y)]
    losses = [quadratic_loss(x, y)]

    for _ in range(epochs):
        grad = grad_func(x, y)
        v = beta * v + (1 - beta) * grad
        x -= eta * v[0]
        y -= eta * v[1]
        path.append((x, y))
        losses.append(quadratic_loss(x, y))
    
    return np.array(path), losses

# %%
#plotting the path
def plot_paths(function, paths, labels, title):
    X, y = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
    Z = function(X, y)

    plt.figure(figsize=(8, 6))
    plt.contour(X, y, Z, levels=50, cmap='jet')

    for path, label in zip(paths, labels):
        plt.plot(path[:, 0], path[:, 1], label=label)
        plt.scatter(path[0, 0], path[0, 1], color='green', label="Start")
        plt.scatter(path[0,0], path[0,1], color='red', label="End")
    
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# %%
eta_bgd = 0.1 # learning rate for BGD
eta_momentum = 0.1 #Learning rate for Momentum
beta = 0.9 #Momentum coefficient
epochs = 50
start_point = (1.5, 1.5) #initial point far from the minimum

# %%
#run optimizations
path_bgd, losses_bgd = batch_gradient_descent(quadratic_grad, eta_bgd, epochs, start_point)
path_momentum, losses_momentum = gradient_descent_momentum(quadratic_grad, eta_momentum, beta, epochs, start_point)

# %%
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(losses_bgd, label="Batch GD")
plt.plot(losses_momentum, label="Momentum GD")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss vs Epochs")

# %%
plot_paths(quadratic_loss, [path_bgd, path_momentum],
           ["Batch Gradient Descent", "Gradient Descent with Momentum"],
           "Oscillations in BGD vs Momentum")


