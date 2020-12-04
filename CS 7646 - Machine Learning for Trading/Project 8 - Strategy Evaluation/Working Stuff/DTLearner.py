"""
-----do not edit anything above this line---

Student Name: Josh Adams (replace with your name)
GT User ID: jadams334 (replace with your User ID)
GT ID: 903475599 (replace with your GT ID)
"""
import os
import sys

import numpy as np
import scipy.stats as scp


class DTLearner(object):
    def __init__(self, leaf_size=5, verbose=False, random_tree=False, **kwargs):
        # region Found on Stackoverflow
        """
        https://stackoverflow.com/questions/8187082/how-can-you-set-class-attributes-from-variable-arguments-kwargs-in-python
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        allowed_keys = {'leaf_size', 'verbose'}
        # initialize all allowed keys to false
        for key, val in kwargs.items():
            if key == 'kwargs':
                kwargs = kwargs["kwargs"]
                break
        for key, val in kwargs.items():
            if key == 'verbose':
                self.verbose = kwargs[key]
            if key == 'leaf_size':
                self.leaf_size = kwargs[key]

        self.tree = np.zeros(shape=())
        self.random_tree = random_tree
        pass  # move along, these aren't the drones you're looking for

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jadams334"  # replace tb34 with your Georgia Tech username

    @staticmethod
    def find_best_feature(data):
        try:
            """
            Suggested by Chris Bohlmann and the Kadanes algorithm
            https://piazza.com/class/kdthusf8jeo7ia?cid=108_f19
            """
            data_X = data[:, :-1]
            data_y = data[:, -1]
            # Suggested way
            second_idx = 0
            max_corr = 0
            feat_idx = 0
            for idx in range(data_X.shape[1]):
                if np.all(data_X[:, idx] == 0):
                    continue
                if np.all(data_X[:, idx] == data_X[0, idx]):
                    continue
                temp_correlation = np.abs(np.corrcoef(data_X[:, idx], data_y)[0, 1])

                if temp_correlation > max_corr:
                    max_corr = temp_correlation
                    second_idx = feat_idx
                    feat_idx = idx
            return feat_idx, second_idx
        except Exception as find_best_feature_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'find_best_feature'", find_best_feature_exception)

    def build_tree(self, data):
        try:
            second_best_i = 0
            # data.shape[1] -1 because data contains the Y column and we do not want to consider that one
            all_column_idx = np.array([str(i) for i in range(data.shape[1] - 1)])
            if data.shape[0] <= self.leaf_size or len(data.shape) == 1:
                return np.array([[-1, scp.mode(data[:, -1]).mode[0], np.nan, np.nan]])
            if np.all(data[:, -1] == data[0, -1]):
                return np.array([[-1, data[0, -1], np.nan, np.nan]])
            else:
                #  Determine best feature to i split on
                if self.random_tree:
                    i = int(np.random.choice(all_column_idx, 1)[0])
                else:
                    i, second_best_i = self.find_best_feature(data=data)

                #  Split_val = data[:,i].median()
                Split_val = np.median(data[:, i])
                split_1 = data[data[:, i] <= Split_val]
                split_2 = data[data[:, i] > Split_val]

                # Check for good splits
                if split_1.size == 0 or split_2.size == 0 or len(split_1.shape) == 1 or len(split_2.shape) == 1:
                    unique_values = np.unique(data[:, i], return_counts=True)[0]
                    if len(unique_values) < 2:
                        return np.array([[-1, data[0, -1], np.nan, np.nan]])
                    else:
                        # Find new splits
                        split_1_idx = np.argwhere(data[:, i] == unique_values[0]).flatten()
                        split_2_idx = np.argwhere(data[:, i] != unique_values[0]).flatten()

                        split_1 = data[split_1_idx]
                        split_2 = data[split_2_idx]

                #  left_tree = build_tree(data[data[:, i] <= Split_val])
                left_tree = self.build_tree(split_1)

                #  right_tree = build_tree(data[data[:, i] > Split_val])
                right_tree = self.build_tree(split_2)

                # root = [i, split_val, 1, lefttree.shape[0] + 1]
                if len(left_tree.shape) == 1:
                    root = np.array([[i, Split_val, 1, 1 + 1]])
                else:
                    root = np.array([[i, Split_val, 1, left_tree.shape[0] + 1]])
                #  return (append(root, lefttree, righttree))
                return np.vstack((root, left_tree, right_tree))
        except Exception as build_tree_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'build_tree'", build_tree_exception)

    def add_evidence(self, data_x, data_y):
        try:
            data = np.hstack((data_x, data_y.reshape(-1, 1)))
            self.tree = self.build_tree(data)
            return
        except Exception as add_evidence_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'add_evidence'", add_evidence_exception)

    def query_single_instance(self, pts):
        try:
            row = 0
            while self.tree[row, 0] != -1:
                feature = int(self.tree[row, 0])
                split_val = self.tree[row, 1]
                if pts[feature] <= split_val:
                    row = row + int(self.tree[row, 2])
                else:
                    row = row + int(self.tree[row, 3])
            return self.tree[row, 1]
        except Exception as query_single_instance_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'query_single_instance'", query_single_instance_exception)

    def query(self, points):
        try:
            results = np.zeros(shape=(points.shape[0],))
            for row in range(points.shape[0]):
                results[row] = self.query_single_instance(points[row, :])
            return results
        except Exception as query_exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'query'", query_exception)


# From LinRegLearner
if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
