"""
A simple wrapper for insane learner.  (c) 2015 Tucker Balch

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

import BagLearner as bl
import LinRegLearner as lrl

class InsaneLearner(object):

    def __init__(self, verbose=False):
        self.learner = bl.BagLearner(learner=bl.BagLearner, kwargs={"learner": lrl.LinRegLearner, "kwargs": {}, "bags": 20, "boost": False, "verbose": verbose}, bags=20, boost=False, verbose=verbose)

    def author(self):
        return "jadams334"

    def addEvidence(self, dataX, dataY):
        self.learner.addEvidence(dataX, dataY)

    def query(self, points):
        return self.learner.query(points)