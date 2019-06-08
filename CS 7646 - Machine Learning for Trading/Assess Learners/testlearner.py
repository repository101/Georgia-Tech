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
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl

plt.style.use("ggplot")     # Adding a style to the plots, makes it look much nicer
np.random.seed(903475599)       # Setting the seed for random to be Student ID, like in previous projects
compare_size = 25


def results(evaluate):
	df = pd.read_csv("./Data/Istanbul.csv", header=0)
	X = df.drop(["date", "EM"], axis=1).values
	Y = df["EM"].values
	train_rows = int(0.6 * X.shape[0])
	trainX = X[:train_rows, :]
	trainY = Y[:train_rows]
	testX = X[train_rows:, :]
	testY = Y[train_rows:]

	for i in evaluate:
		if i == 1:
			try:
				GetPart_A(trainX, trainY, testX, testY)
			except Exception as err1:
				print "Failed attempting to GetPart_A"
				print err1
		if i == 2:
			try:
				GetPart_B(trainX, trainY, testX, testY)
			except Exception as err2:
				print "Failed attempting to GetPart_B"
				print err2
		if i == 3:
			try:
				GetPart_C(X, Y, trainX, trainY, testX, testY)
			except Exception as err3:
				print "Failed attempting to GetPart_C"
				print err3


def GetPart_A(trainX, trainY, testX, testY):
	print "Part A - Starting"
	RMSE_Training = []
	RSME_Testing = []
	leaf_sizes = np.arange(1, compare_size, dtype=np.uint32)
	for leaf_size in leaf_sizes:
		learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
		learner.addEvidence(trainX, trainY)
		pred_y_train = learner.query(trainX)
		rsme_train = math.sqrt(((trainY - pred_y_train) ** 2).sum() / trainY.shape[0])
		RMSE_Training.append(rsme_train)
		pred_y_test = learner.query(testX)
		rsme_test = math.sqrt(((testY - pred_y_test) ** 2).sum() / testY.shape[0])
		RSME_Testing.append(rsme_test)

	fig, ax = plt.subplots()
	pd.DataFrame({
		"Train RMSE": RMSE_Training,
		"Test RMSE": RSME_Testing
	}, index=leaf_sizes).plot(
		ax=ax,
		style="o-",
		title="DTLearner-RMSE VS Leaf-Size"
	)
	plt.xticks(leaf_sizes, leaf_sizes)
	plt.xlabel("Leaf size")
	plt.ylabel("RMSE")
	plt.legend(loc=4)
	plt.tight_layout()
	plt.savefig("Q1.png")

	print "Part A - Finished"
	print ""


def GetPart_B(trainX, trainY, testX, testY):
	print "Part B - Starting"
	try:
		Question_2_Part_1(trainX, trainY, testX, testY)
	except Exception as err2_a:
		print "Failed on Question 2 - Part A"
		print err2_a
	try:
		Question_2_Part_2(trainX, trainY, testX, testY)
	except Exception as err2_b:
		print "Failed on Question 2 - Part B"
		print err2_b
	print "Part B - Finished"
	print ""


def GetPart_C(tempX, tempY, trainX, trainY, testX, testY):
	print "Part C - Starting"
	try:
		Question_3_Part_1(trainX, trainY, testX, testY)
	except Exception as err3_a:
		print "Failed on Question 3 - Part A"
		print err3_a

	try:
		Question_3_Part_2(tempX, tempY)
	except Exception as err3_b:
		print "Failed on Question 3 - Part B"
		print err3_b
	try:
		Question_3_Part_3(tempX, tempY)
	except Exception as err3_c:
		print "Failed on Question 3 - Part C"
		print err3_c
	print "Part C - Finished"
	print ""


def Question_2_Part_1(trainX, trainY, testX, testY):
	print "     B Part 1 - Starting"
	RSME_Training = []
	RSME_Testing = []
	leaf_sizes = np.arange(1, compare_size, dtype=np.uint32)
	for leaf_size in leaf_sizes:
		learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=20, boost=False, verbose=False)
		learner.addEvidence(trainX, trainY)
		train_predY = learner.query(trainX)
		train_rmse = math.sqrt(((trainY - train_predY) ** 2).sum() / trainY.shape[0])
		RSME_Training.append(train_rmse)
		test_predY = learner.query(testX)
		test_rmse = math.sqrt(((testY - test_predY) ** 2).sum() / testY.shape[0])
		RSME_Testing.append(test_rmse)

	fig, ax = plt.subplots()
	pd.DataFrame({
		"Train RMSE": RSME_Training,
		"Test RMSE": RSME_Testing
	}, index=leaf_sizes).plot(
		ax=ax,
		style="o-",
		title="BagLearner-RMSE VS Leaf-Size"
	)
	plt.xticks(leaf_sizes, leaf_sizes)
	plt.xlabel("Leaf size")
	plt.ylabel("RMSE")
	plt.legend(loc=4)
	plt.tight_layout()
	plt.savefig("Q2a.png")
	print "     B Part 1 - Finished"


def Question_2_Part_2(trainX, trainY, testX, testY):
	print "     B Part 2 - Starting"
	train_rmse_values = []
	test_rmse_values = []
	leaf_size_values = np.arange(5, compare_size, dtype=np.uint32)
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
	im = plt.imshow(heatmap, cmap="winter")
	cbar = ax.figure.colorbar(im, ax=ax)
	cbar.ax.set_ylabel("test error", rotation=-90, va="bottom")
	ax.set_xticks(np.arange(len(leaf_size_values)))
	ax.set_xticklabels(leaf_size_values)
	ax.set_yticks(np.arange(len(bag_sizes)))
	ax.set_yticklabels(bag_sizes)
	for i in range(leaf_size_values.shape[0]):
		for j in range(bag_sizes.shape[0]):
			ax.text(i, j, "{:.0f}e-4".format(heatmap[j, i] * 1e4), ha="center", va="center", color="w")
	plt.title("Test Error VS Bag Count & Leaf-Size")
	fig.tight_layout()
	plt.savefig("Q2b.png")
	print "     B Part 2 - Finished"


def Question_3_Part_1(trainX, trainY, testX, testY):
	print "     C Part 1 - Starting"
	RSME_Training = []
	RSME_Testing = []
	Leaf_sizes = np.arange(1, compare_size, dtype=np.uint32)
	for leaf_size in Leaf_sizes:
		learner = rt.RTLearner(leaf_size=leaf_size)
		learner.addEvidence(trainX, trainY)
		train_predY = learner.query(trainX)
		train_rmse = math.sqrt(((trainY - train_predY) ** 2).sum() / trainY.shape[0])
		RSME_Training.append(train_rmse)
		test_predY = learner.query(testX)
		test_rmse = math.sqrt(((testY - test_predY) ** 2).sum() / testY.shape[0])
		RSME_Testing.append(test_rmse)

	fig, ax = plt.subplots()
	pd.DataFrame({
		"Train RMSE": RSME_Training,
		"Test RMSE": RSME_Testing
	}, index=Leaf_sizes).plot(
		ax=ax,
		style="o-",
		title="RMSE of RTLearner against leaf_size"
	)
	plt.xticks(Leaf_sizes, Leaf_sizes)
	plt.xlabel("Leaf size")
	plt.ylabel("RMSE")
	plt.legend(loc=4)
	plt.tight_layout()
	plt.savefig("Q3a.png")
	print "     C Part 1 - Finished"


def Question_3_Part_2(X, Y):
	print "     C Part 2 - Starting"
	# Time comparison
	DecisionTree_Times = []
	RandomDecisionTree_Times = []
	size_values = np.arange(1, X.shape[0], 30, dtype=np.uint64)
	for size_value in size_values:
		trainX = X[:size_value, :]
		trainY = Y[:size_value]
		dt_learner = dt.DTLearner(leaf_size=1)
		start = time.time()
		dt_learner.addEvidence(trainX, trainY)
		end = time.time()
		DecisionTree_Times.append(end-start)
		start = time.time()
		rt_learner = rt.RTLearner(leaf_size=1)
		rt_learner.addEvidence(trainX, trainY)
		end = time.time()
		RandomDecisionTree_Times.append(end-start)

	fig, ax = plt.subplots()
	pd.DataFrame({
		"DTLearner training time": DecisionTree_Times,
		"RTLearner training time": RandomDecisionTree_Times
	}, index=size_values).plot(
		ax=ax,
		style="o-",
		title="Training Time Comparison"
	)
	plt.xlabel("Size of training set")
	plt.ylabel("Time (seconds)")
	plt.legend(loc=2)
	plt.tight_layout()
	plt.savefig("Q3b.png")
	print "     C Part 2 - Finished"


def Question_3_Part_3(X, Y):
	print "     C Part 3 - Starting"
	DecisionTree_Size = []
	RandomDecisionTree_Size = []
	leaf_size_values = np.arange(1, compare_size, dtype=np.uint32)
	for leaf_size in leaf_size_values:
		dt_learner = dt.DTLearner(leaf_size=leaf_size)
		dt_learner.addEvidence(X, Y)
		DecisionTree_Size.append(dt_learner.tree.shape[0])
		rt_learner = rt.RTLearner(leaf_size=leaf_size)
		rt_learner.addEvidence(X, Y)
		RandomDecisionTree_Size.append(rt_learner.tree.shape[0])

	fig, ax = plt.subplots()
	pd.DataFrame({
		"DTLearner size": DecisionTree_Size,
		"RTLearner size": RandomDecisionTree_Size
	}, index=leaf_size_values).plot(
		ax=ax,
		style="o-",
		title="Tree Size co"
	)
	plt.xticks(leaf_size_values, leaf_size_values)
	plt.xlabel("Leaf size")
	plt.ylabel("Size of tree")
	plt.legend(loc=1)
	plt.tight_layout()
	plt.savefig("Q3c1.png")

	DecisionTree_Size = []
	RandomDecisionTree_Size = []
	size_values = np.arange(1, X.shape[0], 30, dtype=np.uint64)
	for size_value in size_values:
		trainX = X[:size_value, :]
		trainY = Y[:size_value]
		dt_learner = dt.DTLearner(leaf_size=1)
		dt_learner.addEvidence(trainX, trainY)
		DecisionTree_Size.append(dt_learner.tree.shape[0])
		rt_learner = rt.RTLearner(leaf_size=1)
		rt_learner.addEvidence(trainX, trainY)
		RandomDecisionTree_Size.append(rt_learner.tree.shape[0])

	fig, ax = plt.subplots()
	pd.DataFrame({
		"DTLearner size": DecisionTree_Size,
		"RTLearner size": RandomDecisionTree_Size
	}, index=size_values).plot(
		ax=ax,
		style="o-",
		title="Tree Size Comparison"
	)
	plt.xlabel("Size of Training Set")
	plt.ylabel("Size of tree")
	plt.legend(loc=1)
	plt.tight_layout()
	plt.savefig("Q3c2.png")
	print "     C Part 3 - Finished"


if __name__ == "__main__":
	questions = [1, 2, 3]
	results(questions)
