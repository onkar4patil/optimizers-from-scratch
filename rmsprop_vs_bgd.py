# rmsprop vs batch gradient descent from scratch
import numpy as np
import matplotlib.pyplot as plt

# %%
#define quadratic loss function
def quadratic_loss(x, y):
    return x**2 + 10 * y**2

# %%
#define gradient of loss function
def quadratic_grad(x, y):
    dx = 2 * x
    dy = 20 * y
    return np.array([dx, dy])

# %%
# Vanilla Gradient Descent
def gradient_descent(grad_func, lr, epochs, start_point):
    x, y = start_point
    path = [(x, y)]
    losses = [quadratic_loss(x, y)]

    for _ in range(epochs):
        grad = grad_func(x, y)
        x -= lr * grad[0]
        y -= lr * grad[1]
        path.append((x, y))
        losses.append(quadratic_loss(x, y))
    
    return np.array(path), losses

# %%
# RMSProp implementation
def rmsprop_optimizer(grad_func, lr, beta, epsilon, epochs, start_point):
    x, y = start_point
    Eg2 = np.array([0.0, 0.0]) #moving average of square of gradients
    path = [(x, y)]
    losses = [quadratic_loss(x, y)]

    for _ in range(epochs):
        grad = grad_func(x ,y) #compute gradients
        Eg2 = beta * Eg2 + (1 - beta) * (grad ** 2) #update moving average

        x -= lr * grad[0] / (np.sqrt(Eg2[0]) + epsilon) #update x
        y -= lr * grad[1] / (np.sqrt(Eg2[1]) + epsilon) #update y

        path.append((x, y))
        losses.append(quadratic_loss(x, y))
    
    return np.array(path), losses

# %%
# Visualization of paths
def plot_paths(function, paths, labels, title):
    X, Y = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
    Z = function(X, Y)

    plt.figure(figsize=(8,6))
    plt.contour(X, Y, Z, levels=50, cmap='jet')

    for path, label in zip(paths, labels):
        plt.plot(path[:,0], path[:,1], label=label)
        plt.scatter(path[0,0], path[0,1], color='green', label="Start")
        plt.scatter(path[0,0], path[0,1], color='red', label="End")
    
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# %%
# Visualizing losses
def plot_losses(losses, labels, title):
    plt.figure(figsize=(8,6))

    for loss, label in zip(losses, labels):
        plt.plot(loss, label=label)
    
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# %%
# Params
lr_gd = 0.1 # learning rate for BGD
lr_rmsprop = 0.1 #learning rate for RMSProp
beta = 0.9 # Decay rate for RMSProp
epsilon = 1e-8
epochs = 100
start_point = (1.5, 1.5) #initial point far from the min

# %%
#Run optimizations
path_gd, losses_gd = gradient_descent(quadratic_grad, lr_gd, epochs, start_point)
path_rmsprop, losses_rmsprop = rmsprop_optimizer(quadratic_grad, lr_rmsprop, beta, epsilon, epochs, start_point)

# %%
#plot results
plot_paths(quadratic_loss, [path_gd, path_rmsprop],
           ["Gradient Descent", "RMSProp"],
           "Optimization Paths: GD vs RMSProp")

plot_losses([losses_gd, losses_rmsprop],
            ["Gradient Descent", "RMSProp"],
            "Loss vs Epochs : GD vs RMSProp")


