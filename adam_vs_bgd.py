# adam optimizer vs batch gradient descent
import numpy as np
import matplotlib.pyplot as plt

# %%
# quadratic loss function
def quadratic_loss(x, y):
    return x**2 + 10 * y **2

# %%
# gradient of loss function
def quadratic_grad(x,y):
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
        path.append((x,y))
        losses.append(quadratic_loss(x,y))
    
    return np.array(path), losses

# %%
# Adam optimizer
def adam_optimizer(grad_func, lr, beta1, beta2, epsilon, epochs, start_point):
    x, y = start_point
    m = np.array([0.0, 0.0]) # First moment (momentum)
    v = np.array([0.0, 0.0]) # second moment (RMSProp)
    path = [(x,y)]
    losses = [quadratic_loss(x,y)]

    for t in range(1, epochs + 1):
        grad = grad_func(x, y) #compute gradient

        #update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad

        #update biased second moment estimate
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        #Updated params
        x -= lr * m_hat[0] / (np.sqrt(v_hat[0]) + epsilon)
        y -= lr * m_hat[1] / (np.sqrt(v_hat[1]) + epsilon)

        path.append((x,y))
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
# plotting loss
def plot_losses(losses, labels, title):
    plt.figure(figsize=(8, 6))

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
lr_adam = 0.1 #learning rate for Adam
beta1 = 0.9 # beta1 for adam
beta2 = 0.999 # beta2 for Adam
epsilon = 1e-8
epochs = 50
start_point = (1.5, 1.5) #initial point far from the min

# %%
#Run optimizations
path_gd, losses_gd = gradient_descent(quadratic_grad, lr_gd, epochs, start_point)
path_adam, losses_adam = adam_optimizer(quadratic_grad, lr_adam, beta1, beta2, epsilon, epochs, start_point)

# %%
#plot results
plot_paths(quadratic_loss, [path_gd, path_adam],
           ["Gradient Descent", "Adam Optimizer"],
           "Optimization Paths: GD vs Adam")

plot_losses([losses_gd, losses_adam],
            ["Gradient Descent", "Adam Optimizer"],
            "Loss vs Epochs : GD vs Adam Optimizer")


