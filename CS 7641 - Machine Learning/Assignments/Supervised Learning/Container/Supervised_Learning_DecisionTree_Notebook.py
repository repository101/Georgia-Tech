#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pathlib
import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from LoadData import LoadData
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import parallel_backend

TESTING = True
DECISION_TREE = False
SUPPORT_VECTOR = True
NEURAL_NET = False
K_NEAREST = False
BOOSTING = False
NORMALIZE_DATA = False
USE_PCA = True
DataSetName = "MNIST"


# In[ ]:


cwd = pathlib.Path().absolute()
if DataSetName == "MNIST":
    training_data_path = "{}/mnist-train-data.csv".format(cwd)
    testing_data_path = "{}/mnist-test-data.csv".format(cwd)
else:
    training_data_path = "{}/fashion-mnist-train-data.csv".format(cwd)
    testing_data_path = "{}/fashion-mnist-test-data.csv".format(cwd)


with parallel_backend('threading'):
    training_labels, training_data, _ = LoadData(training_data_path, normalize=NORMALIZE_DATA)
    testing_labels, testing_data, _ = LoadData(testing_data_path, normalize=NORMALIZE_DATA)

Scaler = StandardScaler().fit(training_data)
        
training_data = Scaler.transform(training_data)
testing_data = Scaler.transform(testing_data)


# In[ ]:


"""
TRAINING TIME

"""


# In[ ]:


solvers = ["Entropy", "Gini"]
classifier_list = []
entropy_runtime = [0.0]
entropy_accuracy = [0.0]

gini_runtime = [0.0]
gini_accuracy = [0.0]


# In[ ]:


with parallel_backend('threading'):
    for solver in solvers:
        clf = DecisionTreeClassifier(criterion=solver.lower(), max_depth=100)
        for i in range(1, 11, 1):
            print("{} - Training Size: {}%".format(solver, (i * 10)))
            start_time = timer()
            with parallel_backend('threading'):
                clf.fit(training_data[:int((60000 * (0.1 * i))), :], training_labels[:int((60000 * (0.1 * i)))])
            end_time = timer()
            elapsed_time = end_time - start_time
            if i == 10:
                classifier_list.append(clf)
            print(elapsed_time)
            if solver == "Entropy":
                entropy_accuracy.append(clf.score(testing_data, testing_labels))
                entropy_runtime.append(elapsed_time)
            else:
                gini_accuracy.append(clf.score(testing_data, testing_labels))
                gini_runtime.append(elapsed_time)


# In[ ]:


entropy_accuracy = np.asarray(entropy_accuracy)
entropy_runtime = np.asarray(entropy_runtime)
gini_accuracy = np.asarray(gini_accuracy)
gini_runtime = np.asarray(gini_runtime)


# In[ ]:


entropy_accuracy.tofile('entropy_accuracy_{}.csv'.format(DataSetName),sep=',',format='%.3f')
entropy_runtime.tofile('entropy_runtime_{}.csv'.format(DataSetName),sep=',',format='%.3f')
gini_accuracy.tofile('gini_accuracy_{}.csv'.format(DataSetName),sep=',',format='%.3f')
gini_runtime.tofile('gini_runtime_{}.csv'.format(DataSetName),sep=',',format='%.3f')


# In[ ]:


"""
Results

"""


# In[ ]:


colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]
solvers = ["Entropy", "Gini"]

run = [entropy_runtime, gini_runtime]
acc = [entropy_accuracy, gini_accuracy]

for solver in range(len(solvers)):
    with plt.style.context('ggplot'):
        fig0, ax0 = plt.subplots()
        ax0.set_xlabel("Percent of Training Set")
        ax0.set_ylabel("Accuracy (%)", color='tab:orange')
        ax0.set_title("Accuracy vs Training Set Size vs Training Time {} \n {}".format(solvers[solver], DataSetName))
        ax0.tick_params(axis='y', labelcolor="black")
        ax0.set_ylim(0, 1.1)
        ax3 = ax0.twinx()
        ax3.set_ylabel("Training Time (s)", color="tab:blue")
        ax3.set_ylim(0, max(max(entropy_runtime), max(gini_runtime)) + 10)
        ax3.tick_params(axis='y', labelcolor="black")
        for i in range(1):        
            ax0.plot([i for i in range(11)], acc[solver], colors[i], marker='o', label=solvers[solver])
            ax3.plot([i for i in range(11)], run[solver], colors[i+1], marker="1", label="{} training-time".format(solvers[solver]))
        fig0.tight_layout()
        directory = "{}/Training_{}_{}_Set_Size_Impact_vs_Training_Time.png".format(cwd, solvers[solver], DataSetName)
        plt.savefig(directory)
#         plt.close("all")

