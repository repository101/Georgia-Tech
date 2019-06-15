"""
template for generating data to fool learners (c) 2016 Tucker Balch
Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Josh Adams (replace with your name)
GT User ID: jadams334 (replace with your User ID)
GT ID: 903475599 (replace with your GT ID)
"""
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

MIN_COLUMNS = 2
MAX_COLUMNS = 10
MIN_ROWS = 10
MAX_ROWS = 1000


def best4LinReg(seed=1489683273):
	# Setting the seed for the random data
	np.random.seed(seed)
	# Randomly generate a N_row X N_col matrix
	X = np.random.rand(np.random.randint(MIN_ROWS, MAX_ROWS + 1), np.random.randint(MIN_COLUMNS, MAX_COLUMNS + 1))

	return X, np.sum(X, axis=1)


def best4DT(seed=1489683273):
	# Setting the seed for the random data
	np.random.seed(seed)

	# Minimum Rows : 10
	# Maximum Rows : 1000
	Number_Of_Rows = np.random.randint(MIN_ROWS, MAX_ROWS + 1)

	# Minimum Columns: 2
	# Maximum Columns: 10
	Number_Of_Columns = np.random.randint(MIN_COLUMNS, MAX_COLUMNS + 1)

	X = np.random.rand(Number_Of_Rows, Number_Of_Columns)
	rand_Column = np.random.randint(0, len(X[1]))

	return X, np.asarray(map(lambda x: 1 if x < 0.5 else 9, X[:, rand_Column]))


def author():
	return 'jadams334'  # Change this to your user ID


if __name__ == "__main__":
	print "There are some who call me... Tim."
