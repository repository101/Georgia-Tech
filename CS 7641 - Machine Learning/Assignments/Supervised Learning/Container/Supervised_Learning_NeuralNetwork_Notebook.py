#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer
from LoadData import LoadData
from sklearn.neural_network import MLPClassifier
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
DataSetName = "Fashion-MNIST"


# In[2]:


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


solvers = ["Adam", "SGD"]

adam_runtime = [0.0]
adam_accuracy = [0.0]
classifier_list = []
sgd_runtime = [0.0]
sgd_accuracy = [0.0]


# In[ ]:


for solver in solvers:
    clf = MLPClassifier(solver=solver.lower(), max_iter=200, verbose=1, hidden_layer_sizes=(100,))
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
        if solver == "SGD":
            sgd_accuracy.append(clf.score(testing_data, testing_labels))
            sgd_runtime.append(elapsed_time)
        else:
            adam_accuracy.append(clf.score(testing_data, testing_labels))
            adam_runtime.append(elapsed_time)


# In[ ]:


adam_accuracy = np.asarray(adam_accuracy)
adam_runtime = np.asarray(adam_runtime)
sgd_accuracy = np.asarray(sgd_accuracy)
sgd_runtime = np.asarray(sgd_runtime)


# In[ ]:


adam_accuracy.tofile('adam_accuracy_{}.csv'.format(DataSetName),sep=',',format='%.3f')
adam_runtime.tofile('adam_runtime_{}.csv'.format(DataSetName),sep=',',format='%.3f')
sgd_accuracy.tofile('sgd_accuracy_{}.csv'.format(DataSetName),sep=',',format='%.3f')
sgd_runtime.tofile('sgd_runtime_{}.csv'.format(DataSetName),sep=',',format='%.3f')


# In[ ]:


"""
Results

"""


# In[ ]:


colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]

run = [sgd_runtime, adam_runtime]
acc = [sgd_accuracy, adam_accuracy]

for solver in range(len(solvers)):
    with plt.style.context('ggplot'):
        fig0, ax0 = plt.subplots()
        ax0.set_xlabel("Percent of Training Set")
        ax0.set_ylabel("Accuracy (%)", color='tab:orange')
        ax0.set_title("Accuracy vs Training Set Size vs Training Time {} \n MNIST".format(solvers[solver]))
        ax0.tick_params(axis='y', labelcolor="black")
        ax0.set_ylim(0, 1.1)
        ax3 = ax0.twinx()
        ax3.set_ylabel("Training Time (s)", color="tab:blue")
        ax3.set_ylim(0, max(max(adam_runtime), max(sgd_runtime)) + 10)
        ax3.tick_params(axis='y', labelcolor="black")
        for i in range(1):        
            ax0.plot([i for i in range(11)], acc[solver], colors[i], marker='o', label=solvers[solver])
            ax3.plot([i for i in range(11)], run[solver], colors[i+1], marker="1", label="{} training-time".format(solvers[solver]))
        fig0.tight_layout()
        directory = "{}/Training_{}_MNIST_Set_Size_Impact_vs_Training_Time.png".format(cwd, solvers[solver])
        plt.savefig(directory)
#         plt.close("all")

