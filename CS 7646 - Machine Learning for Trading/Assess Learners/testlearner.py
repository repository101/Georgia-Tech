""" 			  		 			 	 	 		 		 	  		   	  			  	
Test a learner.  (c) 2015 Tucker Balch

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

import time
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it

plt.style.use("ggplot")     # Adding a style to the plots, makes it look much nicer
np.random.seed(903475599)       # Setting the seed for random to be Student ID, like in previous projects


def test():
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]

    print testX.shape
    print testY.shape

    # create a learner and train it
    # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    learner = dt.DTLearner(leaf_size=1, verbose=False)
    # learner = rt.RTLearner(leaf_size=1, verbose=True)
    # learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 1}, bags=15, boost=False, verbose=False)
    # learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=15, boost=False, verbose=False)
    # learner = it.InsaneLearner(verbose = False)
    learner.addEvidence(trainX, trainY)  # train it
    print learner.author()

    # evaluate in sample
    predY = learner.query(trainX)  # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0, 1]

    # evaluate out of sample
    predY = learner.query(testX)  # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0, 1]


def question_1():
    df = pd.read_csv("./Data/Istanbul.csv", header=0)
    X = df.drop(["date", "EM"], axis=1).values
    Y = df["EM"].values

    train_rows = int(0.6 * X.shape[0])
    trainX = X[:train_rows, :]
    trainY = Y[:train_rows]
    testX = X[train_rows:, :]
    testY = Y[train_rows:]

    train_rmse_values = []
    test_rmse_values = []
    leaf_size_values = np.arange(1, 20+1, dtype=np.uint32)
    for leaf_size in leaf_size_values:
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.addEvidence(trainX, trainY)
        train_predY = learner.query(trainX)
        train_rmse = math.sqrt(((trainY - train_predY) ** 2).sum() / trainY.shape[0])
        train_rmse_values.append(train_rmse)
        test_predY = learner.query(testX)
        test_rmse = math.sqrt(((testY - test_predY) ** 2).sum() / testY.shape[0])
        test_rmse_values.append(test_rmse)

    fig, ax = plt.subplots()
    pd.DataFrame({
        "Train RMSE": train_rmse_values,
        "Test RMSE": test_rmse_values
    }, index=leaf_size_values).plot(
        ax=ax,
        style="o-",
        title="RMSE of DTLearner against leaf_size"
    )
    plt.xticks(leaf_size_values,leaf_size_values)
    plt.xlabel("Leaf size")
    plt.ylabel("RMSE")
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig("Q1.png")


def question_2a():
    df = pd.read_csv("./Data/Istanbul.csv", header=0)
    X = df.drop(["date", "EM"], axis=1).values
    Y = df["EM"].values

    train_rows = int(0.6 * X.shape[0])
    trainX = X[:train_rows, :]
    trainY = Y[:train_rows]
    testX = X[train_rows:, :]
    testY = Y[train_rows:]

    train_rmse_values = []
    test_rmse_values = []
    leaf_size_values = np.arange(1, 20+1, dtype=np.uint32)
    for leaf_size in leaf_size_values:
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=20, boost=False, verbose=False)
        learner.addEvidence(trainX, trainY)
        train_predY = learner.query(trainX)
        train_rmse = math.sqrt(((trainY - train_predY) ** 2).sum() / trainY.shape[0])
        train_rmse_values.append(train_rmse)
        test_predY = learner.query(testX)
        test_rmse = math.sqrt(((testY - test_predY) ** 2).sum() / testY.shape[0])
        test_rmse_values.append(test_rmse)

    fig, ax = plt.subplots()
    pd.DataFrame({
        "Train RMSE": train_rmse_values,
        "Test RMSE": test_rmse_values
    }, index=leaf_size_values).plot(
        ax=ax,
        style="o-",
        title="RMSE of BagLearner against leaf_size"
    )
    plt.xticks(leaf_size_values,leaf_size_values)
    plt.xlabel("Leaf size")
    plt.ylabel("RMSE")
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig("Q2a.png")


def question_2b():
    df = pd.read_csv("./Data/Istanbul.csv", header=0)
    X = df.drop(["date", "EM"], axis=1).values
    Y = df["EM"].values

    train_rows = int(0.6 * X.shape[0])
    trainX = X[:train_rows, :]
    trainY = Y[:train_rows]
    testX = X[train_rows:, :]
    testY = Y[train_rows:]

    train_rmse_values = []
    test_rmse_values = []
    leaf_size_values = np.arange(5, 20+1, dtype=np.uint32)
    bag_sizes = np.arange(10, 50, 10, dtype=np.uint32)
    for bag_size in bag_sizes:
        for leaf_size in leaf_size_values:
            learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=bag_size, boost=False, verbose=False)
            learner.addEvidence(trainX, trainY)
            train_predY = learner.query(trainX)
            train_rmse = math.sqrt(((trainY - train_predY) ** 2).sum() / trainY.shape[0])
            train_rmse_values.append(train_rmse)
            test_predY = learner.query(testX)
            test_rmse = math.sqrt(((testY - test_predY) ** 2).sum() / testY.shape[0])
            test_rmse_values.append(test_rmse)

    heatmap = np.asarray(test_rmse_values).reshape((bag_sizes.shape[0], leaf_size_values.shape[0]))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.grid(False)
    im = plt.imshow(heatmap, cmap="hot")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("test error", rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(leaf_size_values)))
    ax.set_xticklabels(leaf_size_values)
    ax.set_yticks(np.arange(len(bag_sizes)))
    ax.set_yticklabels(bag_sizes)
    for i in range(leaf_size_values.shape[0]):
        for j in range(bag_sizes.shape[0]):
            ax.text(i, j, "{:.0f}e-4".format(heatmap[j, i] * 1e4), ha="center", va="center", color="w")
    plt.title("Test error against number of bag and leaf size")
    fig.tight_layout()
    plt.savefig("Q2b.png")


def question_3a():
    df = pd.read_csv("./Data/Istanbul.csv", header=0)
    X = df.drop(["date", "EM"], axis=1).values
    Y = df["EM"].values

    train_rows = int(0.6 * X.shape[0])
    trainX = X[:train_rows, :]
    trainY = Y[:train_rows]
    testX = X[train_rows:, :]
    testY = Y[train_rows:]

    # RTLearner
    # Compare to question one, in terms of accuracy and sensitivity to overfitting
    # hypothesis: much less sensitive to overfitting because of randomness
    # test RMSE do not decrease much comapred to DTLearner
    train_rmse_values = []
    test_rmse_values = []
    leaf_size_values = np.arange(1, 20 + 1, dtype=np.uint32)
    for leaf_size in leaf_size_values:
        learner = rt.RTLearner(leaf_size=leaf_size)
        learner.addEvidence(trainX, trainY)
        train_predY = learner.query(trainX)
        train_rmse = math.sqrt(((trainY - train_predY) ** 2).sum() / trainY.shape[0])
        train_rmse_values.append(train_rmse)
        test_predY = learner.query(testX)
        test_rmse = math.sqrt(((testY - test_predY) ** 2).sum() / testY.shape[0])
        test_rmse_values.append(test_rmse)

    fig, ax = plt.subplots()
    pd.DataFrame({
        "Train RMSE": train_rmse_values,
        "Test RMSE": test_rmse_values
    }, index=leaf_size_values).plot(
        ax=ax,
        style="o-",
        title="RMSE of RTLearner against leaf_size"
    )
    plt.xticks(leaf_size_values, leaf_size_values)
    plt.xlabel("Leaf size")
    plt.ylabel("RMSE")
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig("Q3a.png")


def question_3b():
    df = pd.read_csv("./Data/Istanbul.csv", header=0)
    X = df.drop(["date", "EM"], axis=1).values
    Y = df["EM"].values

    # Time comparison
    time_values_dt = []
    time_values_rt = []
    size_values = np.arange(1, X.shape[0], 30, dtype=np.uint64)
    for size_value in size_values:
        trainX = X[:size_value, :]
        trainY = Y[:size_value]
        # Measure DTLearner
        dt_learner = dt.DTLearner(leaf_size=1)
        start = time.time()
        dt_learner.addEvidence(trainX, trainY)
        end = time.time()
        time_values_dt.append(end-start)
        # Measure RTLearner
        start = time.time()
        rt_learner = rt.RTLearner(leaf_size=1)
        rt_learner.addEvidence(trainX, trainY)
        end = time.time()
        time_values_rt.append(end-start)

    fig, ax = plt.subplots()
    pd.DataFrame({
        "DTLearner training time": time_values_dt,
        "RTLearner training time": time_values_rt
    }, index=size_values).plot(
        ax=ax,
        style="o-",
        title="Training time comparison on istanbul data"
    )
    plt.xlabel("Size of training set")
    plt.ylabel("Time (seconds)")
    plt.legend(loc=2)
    plt.tight_layout()
    plt.savefig("Q3b.png")


def question_3c():
    df = pd.read_csv("./Data/Istanbul.csv", header=0)
    X = df.drop(["date", "EM"], axis=1).values
    Y = df["EM"].values

    # Space comparison by leaf
    space_values_dt = []
    space_values_rt = []
    leaf_size_values = np.arange(1, 20 + 1, dtype=np.uint32)
    for leaf_size in leaf_size_values:
        # Measure DTLearner
        dt_learner = dt.DTLearner(leaf_size=leaf_size)
        dt_learner.addEvidence(X, Y)
        space_values_dt.append(dt_learner.tree.shape[0])
        # Measure RTLearner
        rt_learner = rt.RTLearner(leaf_size=leaf_size)
        rt_learner.addEvidence(X, Y)
        space_values_rt.append(rt_learner.tree.shape[0])

    fig, ax = plt.subplots()
    pd.DataFrame({
        "DTLearner size": space_values_dt,
        "RTLearner size": space_values_rt
    }, index=leaf_size_values).plot(
        ax=ax,
        style="o-",
        title="Tree size comparison on istanbul data"
    )
    plt.xticks(leaf_size_values, leaf_size_values)
    plt.xlabel("Leaf size")
    plt.ylabel("Size of tree (# of nodes and leaves)")
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig("Q3c1.png")

    # Space comparison by training size
    space_values_dt = []
    space_values_rt = []
    size_values = np.arange(1, X.shape[0], 30, dtype=np.uint64)
    for size_value in size_values:
        trainX = X[:size_value, :]
        trainY = Y[:size_value]
        # Measure DTLearner
        dt_learner = dt.DTLearner(leaf_size=1)
        dt_learner.addEvidence(trainX, trainY)
        space_values_dt.append(dt_learner.tree.shape[0])
        # Measure RTLearner
        rt_learner = rt.RTLearner(leaf_size=1)
        rt_learner.addEvidence(trainX, trainY)
        space_values_rt.append(rt_learner.tree.shape[0])

    fig, ax = plt.subplots()
    pd.DataFrame({
        "DTLearner size": space_values_dt,
        "RTLearner size": space_values_rt
    }, index=size_values).plot(
        ax=ax,
        style="o-",
        title="Tree size comparison on istanbul data"
    )
    plt.xlabel("Size of training set")
    plt.ylabel("Size of tree (# of nodes and leaves)")
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig("Q3c2.png")


if __name__=="__main__":
    #test()
    print "Q1..."; question_1(); print "done."
    print "Q2a..."; question_2a(); print "done."
    print "Q2b..."; question_2b(); print "done." # heatmap
    print "Q3a..."; question_3a(); print "done." # RMSE
    print "Q3b..."; question_3b(); print "done." # Training times
    print "Q3c..."; question_3c(); print "done." # Size of trees
