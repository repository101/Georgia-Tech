#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import parallel_backend
from timeit import default_timer as timer
from LoadData import LoadData


TESTING = True
DECISION_TREE = False
SUPPORT_VECTOR = True
NEURAL_NET = False
K_NEAREST = False
BOOSTING = False
NORMALIZE_DATA = False
USE_PCA = True
DataSetName = "MNIST"


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


# In[3]:


classifier_list = []
runtime = [0.0]
accuracy = [0.0]


# In[ ]:


with parallel_backend('threading'):
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=100, verbose=3)
    for i in range(1, 11, 1):
        print("{} - Training Size: {}%".format("Boosted", (i * 10)))
        start_time = timer()
        with parallel_backend('threading'):
            clf.fit(training_data[:int((60000 * (0.1 * i))), :], training_labels[:int((60000 * (0.1 * i)))])
        end_time = timer()
        elapsed_time = end_time - start_time
        if i == 10:
                classifier_list.append(clf)
        print(elapsed_time)
        accuracy.append(clf.score(testing_data, testing_labels))
        runtime.append(elapsed_time)


# In[ ]:


accuracy = np.asarray(accuracy)
runtime = np.asarray(runtime)


# In[ ]:


accuracy.tofile('boosted_accuracy_{}.csv'.format(DataSetName),sep=',',format='%.3f')
runtime.tofile('boosted_runtime_{}.csv'.format(DataSetName),sep=',',format='%.3f')


# In[ ]:


for i in range(len(classifier_list)):
    disp = plot_confusion_matrix(classifier_list[i], testing_data, testing_labels, values_format=".4g")
    disp.figure_.suptitle("{} Confusion Matrix".format("Boosted"))
    print("TESTING")
    print(disp.confusion_matrix)
    plt.savefig("{}_ConfusionMatrix_{}.png".format("Boosted", DataSetName))


# In[ ]:


"""
Results

"""


# In[ ]:


colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]

run = [runtime]
acc = [accuracy]


with plt.style.context('ggplot'):
    fig0, ax0 = plt.subplots()
    ax0.set_xlabel("Percent of Training Set")
    ax0.set_ylabel("Accuracy (%)", color='tab:orange')
    ax0.set_title("Accuracy vs Training Set Size vs Training Time {} \n {}".format("Boosted", DataSetName))
    ax0.tick_params(axis='y', labelcolor="black")
    ax0.set_ylim(0, 1.1)
    ax3 = ax0.twinx()
    ax3.set_ylabel("Training Time (s)", color="tab:blue")
    ax3.set_ylim(0, max(max(runtime), max(runtime)) + 10)
    ax3.tick_params(axis='y', labelcolor="black")
    for i in range(1):        
        ax0.plot([i for i in range(11)], acc, colors[i], marker='o', label="Boosted Accuracy")
        ax3.plot([i for i in range(11)], run, colors[i+1], marker="1", label="{} training-time".format("Boosted Training Time"))
    fig0.tight_layout()
    directory = "{}/Training_{}_{}_Set_Size_Impact_vs_Training_Time.png".format(cwd, "Boosted", DataSetName)
    plt.savefig(directory)

