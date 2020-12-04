"""
-----do not edit anything above this line---

Student Name: Josh Adams (replace with your name)
GT User ID: jadams334 (replace with your User ID)
GT ID: 903475599 (replace with your GT ID)
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as scp


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
        # self.learners = np.zeros(shape=(self.bags,), dtype=object)
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
        try:
            """
            Add training data to learner

            :param data_x: A set of feature values used to train the learner
            :type data_x: numpy.ndarray
            :param data_y: The value we are attempting to predict given the X data
            :type data_y: numpy.ndarray
            """

            # slap on 1s column so linear regression finds a constant term
            data = pd.concat([data_x, data_y], axis=1).to_numpy()
            subsets = np.random.choice(np.arange(data.shape[0]), size=(self.bags, data.shape[0]), replace=True)

            for i in range(0, self.bags):
                temp_data = data[subsets[i], :]
                bag_x = temp_data[:, :-1]
                bag_y = temp_data[:, -1]
                self.learners[i].add_evidence(bag_x, bag_y)
            return
        except Exception as add_evidence_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'add_evidence'", add_evidence_exception)

    def query(self, points, ):
        try:
            """
            Estimate a set of test points given the model we built.

            :param points: A numpy array with each row corresponding to a specific query.
            :type points: numpy.ndarray
            :return: The predicted result of the input data according to the trained model
            :rtype: numpy.ndarray
            """
            temp_df = pd.DataFrame(data=np.zeros(shape=(self.bags, points.shape[0])))
            for i in range(0, self.bags):
                temp_results = self.learners[i].query(points.to_numpy())
                if self.bags == 1:
                    return temp_results
                else:
                    temp_df.loc[i, :] = temp_results
                    self.results[i] = temp_results
            if self.use_mode:
                return scp.mode(temp_df, axis=0).mode
            else:
                return self.results.mean(axis=0)
        except Exception as query_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'query'", query_exception)
