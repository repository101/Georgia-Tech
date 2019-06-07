"""
A simple wrapper for bag learner.  (c) 2015 Tucker Balch

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

np.random.seed(903475599)       # Setting the seed for random to be student ID


class BagLearner(object):

    def __init__(self, learner, kwargs, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.learners = []
        for _ in xrange(self.bags):
            # Found **kwargs in Piazza, via someones post
            self.learners.append(learner(**kwargs))

    def author(self):
        return "jadams334"

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        for learner in self.learners:
            # Generate the bootstrap data sets
            indices = np.arange(dataX.shape[0])
            # We replace the data since this is a bootstrap dataset
            bootstrap = np.random.choice(indices, size=dataX.shape[0], replace=True)
            # Set XTrain
            Xtrain = dataX[bootstrap, :]
            #Set YTrain
            Ytrain = dataY[bootstrap]
            # Train the learner on the bootstrap data set
            learner.addEvidence(Xtrain, Ytrain)

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        learnerVotes = []
        # Iterate over the learners passed into the bag
        for algorithm in self.learners:
            learnerVotes.append(algorithm.query(points))
        # Return as vstack as it simplifies (atleast for me), processing
        #    the reshape of -1 allows for numpy to make sure the new shape is compatible with the old
        return np.vstack(learnerVotes).mean(axis=0).reshape(-1)


if __name__ == "__main__":
    # Keeping this since we are just coping LinReg
    print "the secret clue is 'zzyzx'"
