#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pathlib

from sklearn import metrics, svm
from sklearn.model_selection import GridSearchCV
from sklearn.utils import parallel_backend

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


# In[3]:


cwd = pathlib.Path().absolute()
training_data_path = "{}/mnist-train-data.csv".format(cwd)
testing_data_path = "{}/mnist-test-data.csv".format(cwd)


training_labels, training_data, _ = LoadData(training_data_path, normalize=NORMALIZE_DATA)
testing_labels, testing_data, _ = LoadData(testing_data_path, normalize=NORMALIZE_DATA)


# In[ ]:


print(training_data.shape)


# In[ ]:


tes = svm.SVC(kernel='rbf', verbose=5, tol=0.1)


# In[ ]:


tes.fit(training_data, training_labels)
y_pred = tes.predict(testing_data)
classification_report = metrics.classification_report(testing_labels, y_pred=y_pred)
print("Classification Report\n {} \n".format(classification_report))


# In[ ]:


confusion_matrix = metrics.plot_confusion_matrix(tes, testing_data, testing_labels, values_format=".4g")
confusion_matrix.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % confusion_matrix.confusion_matrix)


# In[ ]:


GridSearchParameters = {'kernel': ['linear', 'rbf', 'poly'],
                                     'C': [0.001, 0.1, 100, 10000],
                                     'decision_function_shape': ['ovo', 'ovr']}


# In[ ]:


with parallel_backend('threading'):
    svm_clf = GridSearchCV(svm.SVC(), param_grid=GridSearchParameters, scoring='accuracy',
                           verbose=3, cv=2)
    svm_clf.fit(training_data, training_labels)


# In[ ]:





# In[ ]:




