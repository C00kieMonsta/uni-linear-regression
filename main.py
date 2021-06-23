# % PREDICTION OF PROFITS WITH LINEAR REGRESSION %
import numpy as np
import matplotlib.pyplot as plt  # To visualize

import helpers

# 1// Init data
data = np.loadtxt('data/ex1data1.txt', delimiter=',')

m = data.shape[0]
ones = np.ones((m,1))
X_tmp = data[:,0]
X = np.c_[ones, X_tmp]
y = data[:,1].reshape(m,1)

# helpers.show_data(data[:,0], data[:,1], 'Population of City in 10,000s', 'Profit in $10,000s')

# 3// Fit theta with gradient descent
theta_init = np.zeros((2, 1))

theta, history = helpers.gradient_descent(X, y, theta_init, 0.1, 45)

print(history)

# helpers.show_data(list(range(0, 1500)), history, 'Iterations', 'Cost')

# 4// Plot a line from slope and intercept
axes = plt.gca()
x_vals = list(range(0, 45))
y_vals = history
plt.plot(x_vals, y_vals, "--")
plt.show()
# axes = plt.gca()
# x_vals = np.array(axes.get_xlim())
# y_vals = theta[0] + theta[1] * x_vals
# plt.plot(x_vals, y_vals, "--")
# plt.show()
