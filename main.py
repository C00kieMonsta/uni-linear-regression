# % PREDICTION OF PROFITS WITH LINEAR REGRESSION %
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data

import helpers

# 1// Init data
data = np.loadtxt('data/ex1data1.txt', delimiter=',')
m = data.shape[0]
ones = np.ones((m,1))
X = data[:,0].reshape(m,1)
X = np.concatenate((ones, X), axis=1)
y = data[:,1].reshape(m,1)

# 2// Plot raw data
plt.scatter(data[:,0], data[:,1])
plt.xlabel('X'); plt.ylabel('y')
plt.title('Input dataset')

# 3// Fit theta with gradient descent
iterations = 1500
alpha = 0.01
theta_init = np.zeros((2, 1))
theta, history = helpers.gradient_descent(X, y, theta_init, alpha, iterations)

# 4// Plot a line from slope and intercept
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = theta[0] + theta[1] * x_vals
plt.plot(x_vals, y_vals, '--', color='tab:red')
plt.show()
