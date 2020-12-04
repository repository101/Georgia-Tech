"""
-----do not edit anything above this line---

Student Name: Josh Adams (replace with your name)
GT User ID: jadams334 (replace with your User ID)
GT ID: 903475599 (replace with your GT ID)
"""

import numpy as np
import pandas as pd
import scipy.stats as scp


def author(self):
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jadams334"  # replace tb34 with your Georgia Tech username


class BagLearner(object):
    """
    This is a Linear Regression Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, learner=None, bags=20, boost=False, verbose=False, use_mode=True, **kwargs):
        """
        Constructor method
        """
        for key, val in kwargs.items():
            if key == 'kwargs':
                kwargs = kwargs["kwargs"]
                break
        self.learners = np.array([learner(**kwargs) for _ in range(bags)])
        self.bags = bags
        self.boost = boost

        self.results = np.zeros(shape=(self.bags,), dtype=object)
        self.use_mode = use_mode
        # self.verbose = verbose
        pass  # move along, these aren't the drones you're looking for

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jadams334"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        data = pd.concat([data_x, data_y], axis=1).to_numpy()
        subsets = np.random.choice(np.arange(data.shape[0]), size=(self.bags, data.shape[0]), replace=True)

        for i in range(0, self.bags):
            temp_data = data[subsets[i], :]
            bag_x = temp_data[:, :-1]
            bag_y = temp_data[:, -1]
            self.learners[i].add_evidence(bag_x, bag_y)
        return

    def query(self, points, ):
        temp_df = pd.DataFrame(data=np.zeros(shape=(self.bags, points.shape[0])))
        for i in range(0, self.bags):
            temp_results = self.learners[i].query(points.to_numpy())
            if self.bags == 1:
                return temp_results
            else:
                temp_df.loc[i, :] = temp_results
                self.results[i] = temp_results
        if self.use_mode:
            return scp.mode(temp_df, axis=0, nan_policy="omit").mode
        else:
            return self.results.mean(axis=0)
