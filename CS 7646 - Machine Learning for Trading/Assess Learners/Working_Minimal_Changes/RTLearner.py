"""
A simple wrapper for random decision tree.  (c) 2015 Tucker Balch

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

np.random.seed(903475599)


class RTLearner(object):

    def __init__(self, leaf_size=1, verbose = False):
        if leaf_size < 1:
            raise Exception("`leaf_size` must be greater than zero")
        self.leaf_size = leaf_size


    def author(self):
        return "jadams334"


    def build_tree(self, X, Y):
        if self.leaf_size >= X.shape[0]:
            # The region has less than leaf_size data, we return a leaf
            return np.asarray([[-1, np.mean(Y), np.nan, np.nan]])

        if np.unique(Y).shape[0] == 1:
            # The region has data of only one value, we return them all as a leaf
            return np.asarray([[-1, Y[0], np.nan, np.nan]])

        # Find random feature to split on
        best_i = np.random.choice(np.arange(X.shape[1]))

        # Find threshold on which to split
        split_val = np.median(X[:, best_i])
        is_left = X[:, best_i] <= split_val

        if np.median(X[is_left, best_i]) == split_val:
            # If all values of X are the same on this feature, it doesn't make sense to split
            # because the split value (median) is the same. Instead we return this data at a leaf
            return np.asarray([[-1, np.mean(Y), np.nan, np.nan]])

        # Build left tree
        left_tree = self.build_tree(X[is_left], Y[is_left])

        # Build right tree
        right_tree = self.build_tree(X[~is_left], Y[~is_left])

        # Build root
        root = np.asarray([[best_i, split_val, 1, left_tree.shape[0]+1]])

        # Return full tree
        return np.vstack((root, left_tree, right_tree))


    def query_tree(self, tree, X):
        root = tree[0]
        if int(root[0]) == -1:
            # This is a leaf, we return its value
            return root[1]
        elif X[int(root[0])] <= root[1]:
            # Go in the left subtree
            left_tree = tree[int(root[2]):,:]
            return self.query_tree(left_tree, X)
        else:
            # Go in the right subtree
            right_tree = tree[int(root[3]):,:]
            return self.query_tree(right_tree, X)


    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.tree = self.build_tree(dataX, dataY)


    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        Y = []
        for X in points:
            Y.append(self.query_tree(self.tree, X))
        return np.asarray(Y)


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"