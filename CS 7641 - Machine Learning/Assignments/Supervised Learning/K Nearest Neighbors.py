#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import ml_util as utl

from sklearn import metrics
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

Random_Number = 42
TESTING = True


# In[ ]:


gathered_data = utl.setup(["MNIST"])
# gathered_data_fashion = utl.setup(["Fashion-MNIST"])
train_X, train_y, valid_X, valid_y, test_X, test_y = utl.split_data(gathered_data["MNIST"]["X"], gathered_data["MNIST"]["y"])
# fashion_train_X, fashion_train_y, fashion_valid_X, fashion_valid_y, fashion_test_X, fashion_test_y = utl.split_data(gathered_data_fashion["Fashion-MNIST"]["X"], gathered_data_fashion["Fashion-MNIST"]["y"])


# In[ ]:


clf = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')


# In[ ]:


idx = np.arange(1000, 2001, 1000)
print(idx)


# In[ ]:





# In[ ]:





# In[ ]:


results = utl.evaluate_learner(classifier=clf, train_set={"X": train_X, "y": train_y}, 
                               test_set={"X": test_X, "y": test_y}, 
                               validation_set={"X": valid_X, "y": valid_y}, idx=idx, cv=6)


# In[ ]:


results

