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
        theta = theta - (alpha/m)*np.dot(X.T, np.dot(X,theta)-y)
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

def show_data(x, y, x_label='x', y_label='y'):
    plt.scatter(x, y)
    plt.xlabel(x_label); plt.ylabel(y_label)
    plt.show()