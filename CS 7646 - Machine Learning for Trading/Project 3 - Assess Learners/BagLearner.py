# Import allowed per Steven B. Bryant https://piazza.com/class/kdthusf8jeo7ia?cid=109_f2
import numpy as np

import LinRegLearner as lrl


class BagLearner(object):
    """
    This is a Linear Regression Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, learner=None, bags=20, boost=False, insane=False, verbose=False, **kwargs):
        """
        Constructor method
        """

        # self.learner = learner
        self.insane = insane
        if insane:
            # np.array([]) allowed per Steven B. Bryant https://piazza.com/class/kdthusf8jeo7ia?cid=108_f10
            self.learners = np.array([lrl.LinRegLearner(verbose=verbose) for _ in range(bags)])
        else:
            # np.array([]) allowed per Steven B. Bryant https://piazza.com/class/kdthusf8jeo7ia?cid=108_f10

            for key, val in kwargs.items():
                if key == 'kwargs':
                    kwargs = kwargs["kwargs"]
                    break
            self.learners = np.array([learner(**kwargs) for _ in range(bags)])
        self.bags = bags
        self.boost = boost
        # self.learners = np.zeros(shape=(self.bags,), dtype=object)
        self.results = np.zeros(shape=(self.bags,), dtype=object)
        # self.verbose = verbose
        pass  # move along, these aren't the drones you're looking for

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jadams334"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        # slap on 1s column so linear regression finds a constant term
        data = np.hstack((data_x, data_y.reshape(-1, 1)))
        subsets = np.random.choice(np.arange(data.shape[0]), size=(self.bags, data.shape[0]), replace=True)

        for i in range(0, self.bags):
            temp_data = data[subsets[i], :]
            bag_x = temp_data[:, :-1]
            bag_y = temp_data[:, -1]
            self.learners[i].add_evidence(bag_x, bag_y)
        return

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """

        for i in range(0, self.bags):
            temp_results = self.learners[i].query(points)
            if self.bags == 1:
                return temp_results
            else:
                self.results[i] = temp_results
        return self.results.mean(axis=0)
