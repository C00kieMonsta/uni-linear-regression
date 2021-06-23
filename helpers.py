import numpy as np
import matplotlib.pyplot as plt

# Compute cost for linear regression
def compute_cost(X, y, theta=np.array([[0],[0]])):
    m = len(y)
    h = np.dot(X,theta)
    cost = (1.0/(2*m)) * np.dot((h-y).T,(h-y))
    return float(cost)

# Compute gradient descent to fit theta params for linear regression
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        theta_tmp = theta
        cost_history.append(compute_cost(X, y, theta))
        for j in range(len(theta_tmp)):
            h = np.dot(X, theta_tmp)
            x_j = X[:,j].reshape(m,1)
            theta_tmp[j] = theta_tmp[j] - (alpha/m)*np.sum((h - y)*x_j)
        theta = theta_tmp
    return theta, cost_history

def show_data(x, y, x_label='x', y_label='y'):
    plt.scatter(x, y)
    plt.xlabel(x_label); plt.ylabel(y_label)
    plt.show()