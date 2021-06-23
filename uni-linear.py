# % PREDICTION OF PROFITS WITH LINEAR REGRESSION %
import numpy as np
import matplotlib.pyplot as plt  # To visualize

import helpers

# 1// Init data
data = np.loadtxt('data/ex1data1.txt', delimiter=',')

# 2// Format into appropriate shapes
m = data.shape[0]
ones = np.ones((m,1))
X_tmp = data[:,0]
X = np.c_[ones, X_tmp]
y = data[:, 1]

# 3// helper variables
theta_init = np.zeros(2)
iterations = 1500
alpha = 0.01

# 4// Fit theta with gradient descent
theta, history = helpers.gradient_descent(X, y, theta_init, alpha, iterations)

# -- plot raw data with linear regression --
raw = plt.figure(1)
plt.scatter(data[:,0], data[:,1])
plt.xlabel('Population of City in 10,000s'); plt.ylabel('Profit in $10,000s')
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = theta[0] + theta[1] * x_vals
plt.plot(x_vals, y_vals, "--")
raw.show()

# -- plot cost of function --
cost = plt.figure(2)
plt.xlabel('Iterations'); plt.ylabel('Cost')
axes = plt.gca()
x_vals = list(range(0, iterations))
y_vals = history
plt.plot(x_vals, y_vals, "--")
cost.show()

raw_input()