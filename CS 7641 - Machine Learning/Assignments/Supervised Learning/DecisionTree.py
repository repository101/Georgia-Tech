#!/usr/bin/env python
# coding: utf-8

# # Initialize and Setup

# In[1]:


import os
import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier

import ro_ml_util as utl

save_directory = "figures/DecisionTree"
model_name = "Decision Tree"

folders = ["figures/DecisionTree/Complexity_Analysis",
           "figures/DecisionTree/Grid_Search_Results",
           "figures/DecisionTree/Learning_Curves",
           "figures/DecisionTree/Confusion_Matrix",
           "figures/DecisionTree/Metrics"]

directories = {
    "Save Directory": "figures/DecisionTree",
    "Initial Complexity Analysis": "figures/DecisionTree/Initial Complexity Analysis",
    "Grid Search Results": "figures/DecisionTree/Grid Search Results",
    "Learning Curves": "figures/DecisionTree/Learning Curves",
    "Final Complexity Analysis": "figures/DecisionTree/Final Complexity Analysis"
}

Random_Number = 42
TESTING = False
cv = 5
n_jobs = 6
np.random.seed(42)
get_ipython().system('pip install pyarrow')


# In[2]:


gathered_data = utl.setup(["MNIST"])
gathered_data_fashion = utl.setup(["Fashion-MNIST"])
train_X, train_y, valid_X, valid_y, test_X, test_y = utl.split_data(gathered_data["MNIST"]["X"],
                                                                    gathered_data["MNIST"]["y"], normalize=True)
fashion_train_X, fashion_train_y, fashion_valid_X, fashion_valid_y, fashion_test_X, fashion_test_y = utl.split_data(
    gathered_data_fashion["Fashion-MNIST"]["X"],
    gathered_data_fashion["Fashion-MNIST"]["y"],
    normalize=True)


# In[3]:


CHECK_FOLDER = os.path.isdir(save_directory)

# If folder doesn't exist, then create it.
if not CHECK_FOLDER:
    os.makedirs(save_directory)
    print("created folder : ", save_directory)
else:
    print(save_directory, "folder already exists.")

for f in folders:
    if not os.path.isdir(f):
        os.makedirs(f)
        print("created folder : ", f)
    else:
        print(f, "folder already exists.")


# In[4]:


# Criterion ~ entropy' is for information gain

# Max Depth ~ None is default. Can be used for pre pruning the tree

# Min Samples Leaf ~ 1 this is the minimum number of samples required to be a leaf node.


# In[5]:


if TESTING:
    val = 600
    pred_val = 600
    train_sizes = np.linspace(0.05, 1.00, 5)
else:
    val = 4000
    pred_val = 4000
    train_sizes = np.linspace(0.05, 1.00, 20)
print(train_sizes)


# # Initial Model Complexity

# ### Model Complexity Max Depth
# best depth 10

# In[6]:


if TESTING:
    parameter_range = np.arange(1, 5, 1)
else:
    parameter_range = np.arange(1, 40, 1)
    
param_name = "max_depth"
param_name_plot = "Max Depth"
mnist_train_results = None
mnist_test_results = None
fashion_train_results = None
fashion_test_results = None


# In[7]:



start_time = time.time()
for i in range(2):
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        f_name = f"{model_name}_{param_name}_MNIST"
        algorithm_name = f"{model_name} {param_name} MNIST\n Model Complexity"
        plot_title = f"{model_name} MNIST\n Model Complexity"
        ex="a"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        ex="b"
        f_name = f"{model_name}_{param_name}_Fashion_MNIST_{ex}"
        algorithm_name = f"{model_name} {param_name} Fashion MNIST\n Model Complexity"
        plot_title = f"{model_name} Fashion MNIST\n Model Complexity"
    
    temp_train, temp_test = utl.get_model_complexity(classifier=DecisionTreeClassifier(splitter='best'),
                                                     train_X=temp_train_X,
                                                     train_y=temp_train_y, parameter_name=param_name,
                                                     save_dir=save_directory,
                                                     algorithm_name=algorithm_name, parameter_range=parameter_range,
                                                     cv=cv,
                                                     n_jobs=n_jobs, verbose=5, backend='loky',
                                                     param_name_for_plot=param_name_plot,
                                                     f_name=f_name, plot_title=plot_title, is_SVM=False,
                                                     extra_name="Depth_Initial", folder="DecisionTree")
    if i == 0:
        mnist_train_results = temp_train
        mnist_test_results = temp_test
    else:
        fashion_train_results = temp_train
        fashion_test_results = temp_test

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# In[8]:


utl.plot_combined_complexity("DecisionTree", "Maximum Depth", is_NN=False, is_SVM=False,
                             parameter_range=parameter_range, orientation='Horizontal', plt_width=12, plt_height=6,
                             mnist_train_complex=mnist_train_results, mnist_test_complex=mnist_test_results,
                             fashion_train_complex=fashion_train_results, fashion_test_complex=fashion_test_results,
                             folder="DecisionTree", extra_name="Max_Depth_Initial",)
utl.plot_combined_complexity("DecisionTree", "Maximum Depth", is_NN=False, is_SVM=False,
                             parameter_range=parameter_range, orientation='Vertical', plt_height=8,
                             mnist_train_complex=mnist_train_results, mnist_test_complex=mnist_test_results,
                             fashion_train_complex=fashion_train_results, fashion_test_complex=fashion_test_results,
                             folder="DecisionTree", extra_name="Max Depth Initial",)


# ### Model Complexity of Min Samples Leaf:
# 
# best min_samples_leaf = 0.001

# In[9]:


if TESTING:
    parameter_range = 10.** np.arange(-2, 0, 1)
else:
    parameter_range = 10.** np.arange(-5, 0, 1)

a = 10.** np.arange(-5, 0, 1)
parameter_range = np.sort(np.hstack((a, a*7)))
    
param_name = "min_samples_leaf"
param_name_plot = "Minimum Samples for Leaf"
mnist_train_results_leaf = None
mnist_test_results_leaf = None
fashion_train_results_leaf = None
fashion_test_results_leaf = None


# In[10]:



start_time = time.time()
for i in range(2):
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        f_name = f"{model_name}_MNIST_c"
        algorithm_name = f"{model_name} MNIST\n Model Complexity"
        plot_title = f"{model_name} MNIST\n Model Complexity"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        f_name = f"{model_name}_Fashion_MNIST_d"
        algorithm_name = f"{model_name} Fashion MNIST\n Model Complexity"
        plot_title = f"{model_name} Fashion MNIST\n Model Complexity"
    
    temp_train, temp_test = utl.get_model_complexity(classifier=DecisionTreeClassifier(splitter='best'),
                                                     train_X=temp_train_X,
                                                     train_y=temp_train_y, parameter_name=param_name,
                                                     save_dir=save_directory,
                                                     algorithm_name=algorithm_name, parameter_range=parameter_range,
                                                     cv=5, use_log_x=True, 
                                                     n_jobs=n_jobs, verbose=5, backend='loky',
                                                     param_name_for_plot=param_name_plot,
                                                     f_name=f_name, plot_title=plot_title, is_SVM=False,
                                                     extra_name="Min_Leaf_Initial", folder="DecisionTree")
    if i == 0:
        mnist_train_results_leaf = temp_train
        mnist_test_results_leaf = temp_test
    else:
        fashion_train_results_leaf = temp_train
        fashion_test_results_leaf = temp_test
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# In[11]:


utl.plot_combined_complexity("DecisionTree", "Minimum Samples Leaf", is_NN=False, is_SVM=False,
                             parameter_range=parameter_range, orientation='Horizontal', plt_width=12, plt_height=6,
                             mnist_train_complex=mnist_train_results_leaf, mnist_test_complex=mnist_test_results_leaf,
                             fashion_train_complex=fashion_train_results_leaf, extra_name="another_2",
                             fashion_test_complex=fashion_test_results_leaf, folder="DecisionTree")
utl.plot_combined_complexity("DecisionTree", "Minimum Samples Leaf", is_NN=False, is_SVM=False,
                             parameter_range=parameter_range, orientation='Vertical', plt_height=8,
                             mnist_train_complex=mnist_train_results_leaf, mnist_test_complex=mnist_test_results_leaf,
                             fashion_train_complex=fashion_train_results_leaf, extra_name="another",
                             fashion_test_complex=fashion_test_results_leaf, folder="DecisionTree")


# ### Model Complexity of Min Samples Split
# best min_samples_split = 2

# In[12]:


if TESTING:
    parameter_range = np.arange(1, 3, 1)
else:
    parameter_range = np.arange(1, 4, 1)

# p = np.sort(np.hstack((parameter_range, parameter_range*3, parameter_range*7)))
    
param_name = "min_samples_split"
param_name_plot = "Minimum Samples for Split"
mnist_train_results_split = None
mnist_test_results_split = None
fashion_train_results_split = None
fashion_test_results_split = None


# In[13]:


start_time = time.time()
for i in range(2):
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        f_name = f"{model_name}_MNIST_g"
        algorithm_name = f"{model_name} MNIST\n Model Complexity"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        f_name = f"{model_name}_Fashion_MNIST_h"
        algorithm_name = f"{model_name} Fashion MNIST\n Model Complexity"
    
    temp_train, temp_test = utl.get_model_complexity(classifier=DecisionTreeClassifier(splitter='best'),
                                                     train_X=temp_train_X,
                                                     train_y=temp_train_y, parameter_name=param_name,
                                                     save_dir=save_directory,
                                                     algorithm_name=algorithm_name, parameter_range=parameter_range,
                                                     cv=cv,use_log_x=False,
                                                     n_jobs=n_jobs, verbose=5, backend='loky',
                                                     param_name_for_plot=param_name_plot,
                                                     f_name=f_name, plot_title=plot_title, is_SVM=False,
                                                     extra_name="Min_Split_Initial", folder="DecisionTree")
    if i == 0:
        mnist_train_results_split = temp_train
        mnist_test_results_split = temp_test
    else:
        fashion_train_results_split = temp_train
        fashion_test_results_split = temp_test
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# In[14]:


utl.plot_combined_complexity("DecisionTree", "Minimum Samples Split", is_NN=False, is_SVM=False,
                             parameter_range=parameter_range, orientation='Horizontal', plt_width=12, plt_height=6,
                             mnist_train_complex=mnist_train_results_split, mnist_test_complex=mnist_test_results_split,
                             fashion_train_complex=fashion_train_results_split, extra_name="_min_sample_combined_2",
                             fashion_test_complex=fashion_test_results_split, folder="DecisionTree")
utl.plot_combined_complexity("DecisionTree", "Minimum Samples Split", is_NN=False, is_SVM=False,
                             parameter_range=parameter_range, orientation='Vertical', plt_height=8,
                             mnist_train_complex=mnist_train_results_split, mnist_test_complex=mnist_test_results_split,
                             fashion_train_complex=fashion_train_results_split, extra_name="min_sample_combined",
                             fashion_test_complex=fashion_test_results_split, folder="DecisionTree")


# # Initial Confusion Matrix

# In[15]:


clf = DecisionTreeClassifier(max_depth=10, splitter='best', random_state=Random_Number)
clf.fit(train_X.iloc[:val, :], train_y.iloc[:val])
clf2 = DecisionTreeClassifier(max_depth=10, splitter='best', random_state=Random_Number)
clf2.fit(fashion_train_X.iloc[:val, :], fashion_train_y.iloc[:val])


# In[16]:


utl.plot_combined_confusion_matrix(clf, valid_X, valid_y, clf2, fashion_valid_X, fashion_valid_y,
                                   directory=save_directory, fmt=None, plot_width=12, plot_height=6,
                                   folder="DecisionTree", extra_name="Initial")


# # Initial Learning Curve

# In[17]:



start_time = time.time()
results = []
for i in range(1, 2, 1):
    print(f"Working on learning curve: {i}")
    res = {"plt": None,
           "dt_results": None,
           "cv_results": None}
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        title = f"{model_name} MNIST\n Learning Curve"
        f_name = f"{model_name}_MNIST"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        title = f"{model_name} Fashion MNIST\n Learning Curve"
        f_name = f"{model_name}_Fashion_MNIST"
    
    res['dt_results'], res['cv_results'] = utl.plot_learning_curve(
        estimator=DecisionTreeClassifier(random_state=Random_Number, min_samples_split=2, min_samples_leaf=0.001, max_depth=10),
        title=title, train_X=temp_train_X,
        train_y=temp_train_y, cv=cv, f_name=f_name,
        folder="DecisionTree", train_sizes=train_sizes,
        save_individual=True, TESTING=True,
        n_jobs=n_jobs, backend='loky',
        extra_name="InitialLearningCurve")
    results.append(res)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# # Initial Metrics

# In[18]:


clf = DecisionTreeClassifier() # need to change to reflect learning curve results
p, r, f1 = utl.plot_precision_recall(clf, train_X.iloc[:val, :], train_y.iloc[:val], valid_X.iloc[:val, :],
                                 valid_y.iloc[:val], folder="DecisionTree", dataset_name="MNIST",
                                 plot_title="Decision Tree")


# # Gridsearch

# In[19]:


a = 10.** np.arange(-5, 0, 1)

if TESTING:
    all_parameters = {
        'criterion': ['entropy'],
        'min_samples_leaf': np.arange(5, 11, 5),
        'max_depth': np.arange(5, 11, 5)
    }
else:
    all_parameters = {
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': np.sort(np.hstack((a, a*7))),
        'min_samples_split': np.arange(2, 13, 2),
        'max_depth': np.arange(1, 31, 1)
    }


# all_parameters
best_mnist_estimator = None
best_fashion_estimator = None


# In[20]:


start_time = time.time()
for i in range(2):
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        algorithm_name = f"{model_name}_MNIST"
        plot_title = f"{model_name} MNIST\n Model Complexity"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        algorithm_name = f"{model_name}_Fashion_MNIST"
        plot_title = f"{model_name} Fashion MNIST\n Model Complexity"
    
    grid_results_mnist, optimized_dt_mnist = utl.run_grid_search(
        classifier=DecisionTreeClassifier(random_state=Random_Number),
        parameters=all_parameters, train_X=temp_train_X,
        train_y=temp_train_y, cv=cv, extra_name="_a_",
        n_jobs=n_jobs, verbose=5, return_train_score=True,
        refit=True, save_dir=save_directory,
        algorithm_name=algorithm_name, backend='loky', folder="DecisionTree")
    if i == 0:
        best_mnist_estimator = optimized_dt_mnist.best_estimator_
    else:
        best_fashion_estimator = optimized_dt_mnist.best_estimator_

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# # Final Learning Curve

# In[21]:


start_time = time.time()
results = []
for i in range(1, 2, 1):
    print(f"Working on learning curve: {i}")
    res = {"plt": None,
           "dt_results": None,
           "cv_results": None}
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        title = f"{model_name} MNIST\n Learning Curve"
        f_name = f"{model_name}_MNIST"
        optimized_dt = best_mnist_estimator
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        title = f"{model_name} Fashion MNIST\n Learning Curve"
        f_name = f"{model_name}_Fashion_MNIST"
        optimized_dt = best_fashion_estimator
    
    res['dt_results'], res['cv_results'] = utl.plot_learning_curve(estimator=optimized_dt,
                                                                   title=title, train_X=temp_train_X,
                                                                   train_y=temp_train_y, cv=cv, f_name=f_name,
                                                                   folder="DecisionTree", train_sizes=train_sizes,
                                                                   save_individual=True, TESTING=True,
                                                                   n_jobs=n_jobs, backend='loky',
                                                                   extra_name="Final_Learning_Curve")
    results.append(res)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# # Final Confusion Matrix

# In[22]:


utl.plot_combined_confusion_matrix(best_mnist_estimator, test_X, test_y, best_fashion_estimator, fashion_test_X,
                                   fashion_test_y,
                                   directory=save_directory, fmt=None, plot_width=12, plot_height=6,
                                   extra_name="Decision_conf_Tree_Final", folder="DecisionTree")


# # Final Metrics

# In[23]:


p, r, f1 = utl.plot_precision_recall(best_mnist_estimator, train_X.iloc[:val, :], train_y.iloc[:val], test_X.iloc[:val, :],
                                     test_y.iloc[:val], folder="DecisionTree", dataset_name="MNIST",
                                     plot_title="Decision Tree Test Set", is_final=True)


# In[24]:


p, r, f1 = utl.plot_precision_recall(best_fashion_estimator, fashion_train_X.iloc[:val, :], fashion_train_y.iloc[:val], 
                                     fashion_test_X.iloc[:val, :],
                                 fashion_test_y.iloc[:val], folder="DecisionTree", dataset_name="Fashion MNIST",
                                 plot_title="Decision Tree Test Set", is_final=True)


# In[25]:


best_mnist_estimator

