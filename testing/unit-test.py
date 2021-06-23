import unittest
import numpy as np
import sys
import os

path_to_folder = '/Users/antoineboxho/Developer/personal/learnings/python/machine-learning/playground/uni-linear-regression'

sys.path.append(path_to_folder)

import helpers

class TestHelpers(unittest.TestCase):

    # 1// Init data
    data = np.loadtxt(os.path.join(path_to_folder, 'testing/test-data/ex1data1.txt'), delimiter=',')
    m = data.shape[0]
    ones = np.ones((m,1))
    y = data[:,1].reshape(m,1)
    X = data[:,0].reshape(m,1)
    X = np.concatenate((ones, X), axis=1)

    def test_cost_function(self):
        np.testing.assert_almost_equal(helpers.compute_cost(self.X, self.y, np.array([[-1],[2]])), 54.24, decimal=2)
        np.testing.assert_almost_equal(helpers.compute_cost(self.X, self.y, np.array([[0],[0]])), 32.07, decimal=2)

    def test_gradient_descent_function(self):
        theta, J_history = helpers.gradient_descent(self.X, self.y, np.zeros((2, 1)), 0.01, 1500)
        np.testing.assert_almost_equal(theta[0], -3.6303, decimal=4)
        np.testing.assert_almost_equal(theta[1], 1.1664, decimal=4)

if __name__ == '__main__':
    unittest.main()
