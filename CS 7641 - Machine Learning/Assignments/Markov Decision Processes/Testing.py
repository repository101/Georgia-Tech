import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

if __name__ == "__main__":
    X = np.asarray([[0.5403, -0.4161],
                    [-0.9900, -0.6536],
                    [0.2837, 0.9602]])

    y = np.asarray([[4], [2], [3]])
    Theta1 = np.asarray([[0.1, 0.3, 0.5],
                         [0.2, 0.4, 0.6]])
    Theta2 = np.asarray([[0.7, 1.1, 1.5],
                         [0.8, 1.2, 1.6],
                         [0.9, 1.3, 1.7],
                         [1.0, 1.4, 1.8]])
    m = 3
    eye_matrix = np.eye(4)
    y_matrix = np.asarray([[0, 0, 0, 1],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]])
    temp = np.ones(shape=(3, 1))
    A_1 = np.hstack((temp, X))

    Z_2 = A_1.dot(Theta1.transpose())
    A_2 = sigmoid(Z_2)
    A_2_with_bias = np.hstack((np.ones(shape=(A_2.shape[0], 1)), A_2))

    Z_3 = A_2_with_bias.dot(Theta2.transpose())
    A_3 = sigmoid(Z_3)
    unregularized_cost_part1 = -y_matrix.transpose().dot(np.log(A_3))
    unregularized_cost_part2 = (1-y_matrix.transpose()).dot(np.log(1-A_3))
    unregularized_cost_temp = (eye_matrix * (unregularized_cost_part1 - unregularized_cost_part2))
    unregularized_cost = (1/m) * np.sum(np.sum(unregularized_cost_temp))
    print("Hey")
