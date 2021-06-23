# % PREDICTION OF HOUSE PRICES WITH LINEAR REGRESSION %
import numpy as np
import matplotlib.pyplot as plt  # To visualize

import helpers

# 1// Init data
data = np.loadtxt('data/ex1data2.txt', delimiter=',')

# 2// Format into appropriate shapes
y = data[:, 2]
m = y.size
ones = np.ones((m,1))
X_tmp = helpers.feature_normalization(data[:,:2])
X = np.c_[ones, X_tmp]

helpers.show_data(data[:,0], data[:,2], 'sq meters', 'price house')
helpers.show_data(data[:,1], data[:,2], '# of rooms', 'price house')

# 3// helper variables
theta_init = np.zeros(3)
iterations = 400
alpha = 0.1

# 4// Fit theta with gradient descent
theta, history = helpers.gradient_descent(X, y, theta_init, alpha, iterations)

prediction = theta[0] + theta[1]*1650 + theta[2]*3
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(float(prediction)))

