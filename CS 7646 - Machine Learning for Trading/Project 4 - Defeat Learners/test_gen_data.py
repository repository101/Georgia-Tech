import math

import numpy as np

import DTLearner as dt
import LinRegLearner as lrl
from gen_data import best_4_dt, best_4_lin_reg


def compare_os_rmse(learner1, learner2, x, y):
    """
    Compares the out-of-sample root mean squared error of your LinRegLearner and DTLearner.

    :param learner1: An instance of LinRegLearner
    :type learner1: class:'LinRegLearner.LinRegLearner'
    :param learner2: An instance of DTLearner
    :type learner2: class:'DTLearner.DTLearner'
    :param x: X data generated from either gen_data.best_4_dt or gen_data.best_4_lin_reg
    :type x: numpy.ndarray
    :param y: Y data generated from either gen_data.best_4_dt or gen_data.best_4_lin_reg
    :type y: numpy.ndarray
    :return: The root mean squared error of each learner
    :rtype: tuple
    """
    # compute how much of the data is training and testing
    train_rows = int(math.floor(0.6 * x.shape[0]))
    test_rows = x.shape[0] - train_rows
    
    # separate out training and testing data
    train = np.random.choice(x.shape[0], size=train_rows, replace=False)
    test = np.setdiff1d(np.array(range(x.shape[0])), train)
    train_x = x[train, :]
    train_y = y[train]
    test_x = x[test, :]
    test_y = y[test]
    
    # train the learners
    learner1.add_evidence(train_x, train_y)  # train it
    learner2.add_evidence(train_x, train_y)  # train it
    
    # evaluate learner1 out of sample
    pred_y = learner1.query(test_x)  # get the predictions
    rmse1 = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    
    # evaluate learner2 out of sample
    pred_y = learner2.query(test_x)  # get the predictions
    rmse2 = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    
    return rmse1, rmse2


def test_lr(seed):
    # create two learners and get data
    lrlearner = lrl.LinRegLearner(verbose=False)
    dtlearner = dt.DTLearner(verbose=False, leaf_size=1)
    x, y = best_4_lin_reg(seed=seed)
    
    # compare the two learners
    rmse_lr, rmse_dt = compare_os_rmse(lrlearner, dtlearner, x, y)
    
    # share results
    print()
    print("best_4_lin_reg() results")
    print(f"RMSE LR    : {rmse_lr}")
    print(f"RMSE DT    : {rmse_dt}")
    if rmse_lr < 0.9 * rmse_dt:
        return 1
    else:
        return -1


def test_dt(seed):
    lrlearner = lrl.LinRegLearner(verbose=False)
    dtlearner = dt.DTLearner(verbose=False, leaf_size=1)
    x, y = best_4_dt()
    
    # compare the two learners
    rmse_lr, rmse_dt = compare_os_rmse(lrlearner, dtlearner, x, y)
    
    # share results
    print()
    print("best_4_dt() results")
    print(f"RMSE LR    : {rmse_lr}")
    print(f"RMSE DT    : {rmse_dt}")
    if rmse_dt < 0.9 * rmse_lr:
        print("DT < 0.9 LR:  pass")
    else:
        print("DT >= 0.9 LR:  fail")
    print


if __name__ == '__main__':
    results = np.zeros(shape=(10000,))
    
    for i in range(10000):
        results[i] = test_lr()
