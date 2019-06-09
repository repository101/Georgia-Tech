"""
A simple wrapper for decision tree.  (c) 2015 Tucker Balch

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
"""
# Student Name: Josh Adams (replace with your name)
# GT User ID: jadams334 (replace with your User ID)
# GT ID: 903475599 (replace with your GT ID)

import numpy as np


class DTLearner(object):

	def __init__(self, leaf_size=1, verbose=False):
		# TODO: Possibly remove self.tree as it is causing issues
		self.tree = []
		if leaf_size < 1:
			raise Exception("`leaf_size` must be greater than zero")
		self.leaf_size = leaf_size

	def author(self):
		return "jadams334"

	def build_tree(self, X, Y):
		# Establish that -1 denotes a leaf

		# From 'CS 7646: How to build a decision tree from data.{pdf/slides}'
		if self.leaf_size >= X.shape[0]:
			# Return as leaf
			return np.asarray([[-1, np.mean(Y),
								np.nan, np.nan]])

		if np.unique(Y).shape[0] == 1:
			# Return as leaf
			return np.asarray([[-1, Y[0],
								np.nan, np.nan]])

		data_correlation = []

		# Determine best feature i to split on
		for i in xrange(X.shape[1]):
			# Getting the variance of the data using numpy
			data_variance = np.var(X[:, i])
			# Getting the correlation coefficient using numpy
			data_correlation_coefficient = np.corrcoef(X[:, i], Y)[0, 1] if data_variance > 0 else 0
			data_correlation.append(data_correlation_coefficient)

		split_feature = np.argsort(data_correlation)[::-1][0]
		split_val = np.median(X[:, split_feature])

		if np.median(X[X[:, split_feature] <= split_val, split_feature]) == split_val:
			# Check if all values are same then return leaf if true
			return np.asarray([[-1, np.mean(Y), np.nan, np.nan]])

		# Call build to build left_child
		left_child = self.build_tree(X[X[:, split_feature] <= split_val],
									Y[X[:, split_feature] <= split_val])

		# Call build to build right_child
		right_child = self.build_tree(X[X[:, split_feature] > split_val],
									 Y[X[:, split_feature] > split_val])

		# Establish the root of the tree
		# From 'CS 7646: How to build a decision tree from data.{pdf/slides}'
		root = np.asarray([[split_feature,
							split_val,
							1,
							left_child.shape[0] + 1]])

		# Return as vstack (vertical stack) as it simplifies (atleast for me), processing.
		return np.vstack((root, left_child, right_child))

	def query_tree(self, tree, data):
		# This was a hard concept for me to grasp
		# Tree [[root],[value],[left_child],[right_child]]
		tree_root = tree[0]
		if int(tree_root[0]) == -1:
			# If the value is -1 we know it is a leaf so we should just return the value
			return tree_root[1]

		elif data[int(tree_root[0])] <= tree_root[1]:
			# Go left and process recursively
			left_child = tree[
						 int(tree_root[2]):, :]
			return self.query_tree(left_child, data)

		else:
			# Go right and process recursively
			right_child = tree[
						  int(tree_root[3]):, :]
			return self.query_tree(right_child, data)

	def addEvidence(self, dataX, dataY):
		"""
		@summary: Add training data to learner
		@param dataX: X values of data to add
		@param dataY: the Y training values
		"""
		# Train the decision tree (build decision tree)
		#   I defined this in __init__ but was unsuccessful in getting it to populate without a
		#   direct call to self.build
		self.tree = self.build_tree(dataX, dataY)

	def query(self, data):
		"""
		@summary: Estimate a set of test points given the model we built.
		@param data: should be a numpy array with each row corresponding to a specific query.
		@returns the estimated values according to the saved model.
		"""
		result = []
		for data_point in data:
			# Iterate over the points and query the tree with the point then add to result
			result.append(self.query_tree(self.tree, data_point))
		return np.asarray(result)  # Return the results as a numpy array


if __name__ == "__main__":
	print "the secret clue is 'zzyzx'"
