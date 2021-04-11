import numpy as np



# Compute cost for linear regression
def compute_cost(X, Y, theta=np.array([[0],[0]])):
    m = len(Y)
    k = 1/(2*m)
    h = np.dot(X, theta) # mx2 . 2x1  = mx1
    J = k * np.sum((h - Y) ** 2) # mx1 - mx1 = mx1
    return J

# Compute gradient descent to fit theta params for linear regression
def gradient_descent(X, Y, theta, alpha, iterations):
    m = len(Y)
    J_history = []
    for i in range(iterations):
        h = np.matmul(X, theta)
        theta = theta - alpha * (1 / m) * np.transpose(np.matmul(np.transpose(h - Y), X))
        J_history.append(compute_cost(X, Y, theta))

    return theta, J_history
