from scipy import stats
import numpy as np


# This method computes entropy for information gain
def entropy(class_y):
    # Input:
    #   class_y         : list of class labels (0's and 1's)

    # TODO: Compute the entropy for a list of classes
    #
    # Example:
    #    entropy([0,0,0,1,1,1,1,1,1]) = 0.92
    try:
        unique, counts = np.unique(class_y, return_counts=True)
        count_dict = dict(zip(unique, counts))
    except Exception as ex:
        print("An exception occurred during the entropy function: \n{}".format(ex))

    if 0 in class_y:
        n_zero = count_dict[0]
    else:
        n_zero = 0

    if 1 in class_y:
        n_one = count_dict[1]
    else:
        n_one = 0

    if (n_zero + n_one) == 0:
        p_zero = 0
        zero_side = 0
        p_one = 0
        one_side = 0
    else:
        p_zero = n_zero / (n_zero + n_one)
        p_one = n_one / (n_zero + n_one)
        if n_zero == 0:
            zero_side = 0
        else:
            zero_side = np.log2(p_zero)
        if n_one == 0:
            one_side = 0
        else:
            one_side = np.log2(p_one)

    entropy = (-p_zero * zero_side) - (p_one * one_side)
    if (entropy == np.nan) or (entropy <= 0):
        entropy = 0

    return entropy


def partition_classes(X, y, split_attribute, split_val):
    # Inputs:
    #   X               : data containing all attributes
    #   y               : labels
    #   split_attribute : column index of the attribute to split on
    #   split_val       : either a numerical or categorical value to divide the split_attribute

    # TODO: Partition the data(X) and labels(y) based on the split value - BINARY SPLIT.
    # 
    # You will have to first check if the split attribute is numerical or categorical    
    # If the split attribute is numeric, split_val should be a numerical value
    # For example, your split_val could be the mean of the values of split_attribute
    # If the split attribute is categorical, split_val should be one of the categories.   
    #
    # You can perform the partition in the following way
    # Numeric Split Attribute:
    #   Split the data X into two lists(X_left and X_right) where the first list has all
    #   the rows where the split attribute is less than or equal to the split value, and the 
    #   second list has all the rows where the split attribute is greater than the split 
    #   value. Also create two lists(y_left and y_right) with the corresponding y labels.
    #
    # Categorical Split Attribute:
    #   Split the data X into two lists(X_left and X_right) where the first list has all 
    #   the rows where the split attribute is equal to the split value, and the second list
    #   has all the rows where the split attribute is not equal to the split value.
    #   Also create two lists(y_left and y_right) with the corresponding y labels.

    '''
    Example:
    
    X = [[3, 'aa', 10],                 y = [1,
         [1, 'bb', 22],                      1,
         [2, 'cc', 28],                      0,
         [5, 'bb', 32],                      0,
         [4, 'cc', 32]]                      1]
    
    Here, columns 0 and 2 represent numeric attributes, while column 1 is a categorical attribute.
    
    Consider the case where we call the function with split_attribute = 0 and split_val = 3 (mean of column 0)
    Then we divide X into two lists - X_left, where column 0 is <= 3  and X_right, where column 0 is > 3.
    
    X_left = [[3, 'aa', 10],                 y_left = [1,
              [1, 'bb', 22],                           1,
              [2, 'cc', 28]]                           0]
              
    X_right = [[5, 'bb', 32],                y_right = [0,
               [4, 'cc', 32]]                           1]

    Consider another case where we call the function with split_attribute = 1 and split_val = 'bb'
    Then we divide X into two lists, one where column 1 is 'bb', and the other where it is not 'bb'.
        
    X_left = [[1, 'bb', 22],                 y_left = [1,
              [5, 'bb', 32]]                           0]
              
    X_right = [[3, 'aa', 10],                y_right = [1,
               [2, 'cc', 28],                           0,
               [4, 'cc', 32]]                           1]
               
    '''


    # Check if X is an numpy.ndarray, if it is we pass, otherwise we convert to a numpy array
    if isinstance(X, np.ndarray):
        pass
    else:
        X = np.asarray(X)
    # Check if y is an numpy.ndarray, if it is we pass, otherwise we convert to a numpy array
    if isinstance(y, np.ndarray):
        pass
    else:
        y = np.asarray(y)

    # Get the size, unique values and count of unique values for y
    y_size = y.size
    y_unique, y_unique_count = np.unique(y, return_counts=True)
    total = y_unique_count[0]+y_unique_count[1]
    # Calculate percent different to be a measure of when to end recursion and establish a leaf
    percent_diff = abs((y_unique_count[0]/total) - (y_unique_count[1]/total))
    if (y_size < 10) or (len(y_unique) <= 1) or (percent_diff > 0.95):
        return y_unique[0]

    X_left = []
    X_right = []

    y_left = []
    y_right = []
    data_type_numeric = False
    data_type_categorical = False

    if isinstance(split_val, type("String")):
        data_type_categorical = True
    elif isinstance(split_val, (np.int16, np.int32, np.int64, np.float16,
                                np.float32, np.float64, type(3), type(3.1))):
        data_type_numeric = True
    else:
        print("Data type unknown:   util.py - ln136")

    if data_type_numeric and not data_type_categorical:
        new_X = X[:, split_attribute].astype(int)
        X_left_index = np.where(new_X <= split_val)
        X_left = X[X_left_index]
        X_right_index = np.where(new_X > split_val)
        X_right = X[X_right_index]
        y_left = y[X_left_index]
        y_right = y[X_right_index]
    elif data_type_categorical and not data_type_numeric:
        new_X = X[:, split_attribute]
        X_left_index = np.where(new_X == split_val)
        X_left = X[X_left_index]
        X_right_index = np.where(new_X != split_val)
        X_right = X[X_right_index]
        y_left = y[X_left_index]
        y_right = y[X_right_index]
    else:
        print("Data Split Failed:   util.py - ln155")

    return X_left, X_right, np.reshape(y_left, (-1, 1)), np.reshape(y_right, (-1, 1))


def information_gain(previous_y, current_y):
    # Inputs:
    #   previous_y: the distribution of original labels (0's and 1's)
    #   current_y:  the distribution of labels after splitting based on a particular
    #               split attribute and split value

    # TODO: Compute and return the information gain from partitioning the previous_y labels
    # into the current_y labels.
    # You will need to use the entropy function above to compute information gain
    # Reference: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf

    """
    Example:

    previous_y = [0,0,0,1,1,1]
    current_y = [[0,0], [1,1,1,0]]

    info_gain = 0.45915
    """

    H = entropy(previous_y)
    H_L = entropy(current_y[0])
    H_R = entropy(current_y[1])
    total = len(current_y[0]) + len(current_y[1])
    info_gain = H - (H_L * (len(current_y[0]) / total) + H_R * (len(current_y[1]) / total))

    return info_gain

