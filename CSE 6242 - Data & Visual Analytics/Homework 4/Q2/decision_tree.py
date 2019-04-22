from util import entropy, information_gain, partition_classes
from scipy import stats
import numpy as np
import ast


class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        self.tree = {'left_child': None, 'right_child': None}
        self.label = None
        self.split_column = None
        self.split_value = None

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree

        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        if isinstance(X, np.ndarray):
            pass
        else:
            X = np.asarray(X)
        if isinstance(y, np.ndarray):
            pass
        else:
            y = np.asarray(y)

        y_size = y.size
        y_unique, y_unique_count = np.unique(y, return_counts=True)
        if len(y_unique) > 1:
            total = y_unique_count[0] + y_unique_count[1]
            # Calculate percent different to be a measure of when to end recursion and establish a leaf
            percent_diff = abs((y_unique_count[0] / total) - (y_unique_count[1] / total))

        if (y_size < 10) or (len(y_unique) <= 1) or (percent_diff > 0.95):
            if len(y_unique) > 1:
                if y_unique_count[0] > y_unique_count[1]:
                    self.label = 0
                elif y_unique_count[1] > y_unique_count[0]:
                    self.label = 1
                else:
                    self.label = np.random.randint(0, 2)
            else:
                self.label = y_unique[0]

        num_of_columns = X[0].size
        list_of_all_column_performance = []
        best_split_value = 0
        most_info_gain = 0
        best_split_attribute = 0
        best_x_left = 0
        split_name = 0
        best_x_right = 0
        best_y_left = 0
        best_y_right = 0

        for column in range(num_of_columns):
            temp_subset = X[:, column]

            if isinstance(temp_subset[0], type("str")):
                data_dict = {
                    'mode': stats.mode(temp_subset)[0][0]
                }
            else:
                data_dict = {
                    'mode': stats.mode(temp_subset)[0][0],
                    'median': np.median(temp_subset),
                    'mean': np.mean(temp_subset),
                    '25thQuantile': np.quantile(temp_subset, 0.25),
                    '50thQuantile': np.quantile(temp_subset, 0.5),
                    '75thQuantile': np.quantile(temp_subset, 0.75)
                }

            new_y_unique = np.unique(y)
            if len(new_y_unique) <= 1:
                self.label = new_y_unique[0]
                return
            for key, temp_value in data_dict.items():
                if (X is None) or (y is None) or (column is None) or (temp_value is None):
                    return
                try:
                    X_left, X_right, y_left, y_right = partition_classes(X, y, column, temp_value)
                except Exception as err:
                    unique, unique_count = np.unique(y, return_counts=True)
                    if unique_count[0] > unique_count[1]:
                        self.label = unique_count[0]
                    elif unique_count[1] > unique_count[0]:
                        self.label = unique_count[1]
                    else:
                        return
                    return
                prev_y = np.concatenate((y_left, y_right), axis=0)
                current_y = [y_left, y_right]
                temp_info_gain = information_gain(prev_y, current_y)
                if temp_info_gain > most_info_gain:
                    most_info_gain = temp_info_gain
                    best_x_left = X_left
                    split_name = key
                    best_x_right = X_right
                    best_y_left = y_left
                    best_y_right = y_right
                    best_split_value = temp_value
                    best_split_attribute = column
            best_data_split = [best_x_left, best_x_right, best_y_left, best_y_right]
            list_of_all_column_performance.append([best_split_attribute, split_name, best_split_value,
                                                   most_info_gain, best_data_split, data_dict])

        list_of_all_column_performance.sort(key=lambda feature: feature[3], reverse=True)
        column_to_split_on = list_of_all_column_performance[0][0]
        value_to_split_on = list_of_all_column_performance[0][2]
        self.split_column = column_to_split_on
        self.split_value = value_to_split_on

        subtree_left = DecisionTree()
        subtree_right = DecisionTree()
        subtree_left.learn(best_x_left, best_y_left)
        subtree_right.learn(best_x_right, best_y_right)
        self.tree['left_child'] = subtree_left
        self.tree['right_child'] = subtree_right

    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        # If the label is not none, return that as the label because we only set the label when at a leaf
        if self.label is not None:
            return int(self.label)
        elif record[self.split_column] <= self.split_value:
            return self.tree['left_child'].classify(record)
        else:
            return self.tree['right_child'].classify(record)

