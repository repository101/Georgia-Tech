#!/usr/bin/env python
# coding: utf-8

# # Initialize and Setup

# In[ ]:


import os
import time

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import ro_ml_util as utl

save_directory = "figures/Boosting"
model_name = "Boosting"

folders = ["figures/Boosting/Complexity_Analysis",
           "figures/Boosting/Grid_Search_Results",
           "figures/Boosting/Learning_Curves",
           "figures/Boosting/Confusion_Matrix",
           "figures/Boosting/Metrics"]

directories = {
    "Save Directory": "figures/Boosting",
    "Initial Complexity Analysis": "figures/Boosting/Initial Complexity Analysis",
    "Grid Search Results": "figures/Boosting/Grid Search Results",
    "Learning Curves": "figures/Boosting/Learning Curves",
    "Final Complexity Analysis": "figures/Boosting/Final Complexity Analysis"
}

Random_Number = 42
TESTING = False
cv = 5
n_jobs = -1
np.random.seed(42)
get_ipython().system('pip install pyarrow')


# In[ ]:


gathered_data = utl.setup(["MNIST"])
gathered_data_fashion = utl.setup(["Fashion-MNIST"])
train_X, train_y, valid_X, valid_y, test_X, test_y = utl.split_data(gathered_data["MNIST"]["X"],
                                                                    gathered_data["MNIST"]["y"], normalize=True)
fashion_train_X, fashion_train_y, fashion_valid_X, fashion_valid_y, fashion_test_X, fashion_test_y = utl.split_data(
    gathered_data_fashion["Fashion-MNIST"]["X"],
    gathered_data_fashion["Fashion-MNIST"]["y"],
    normalize=True)


# In[ ]:


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


# In[ ]:


# Base Estimator ~ default is None, and will be a decision tree with max depth 1. Experiments changing the base 
#   estimators max depth from 1:5

# N Estimators ~ default is 50. Experiment with np.arange(10, 501, 10)

if TESTING:
    val = 600
    pred_val = 600
    train_sizes = np.linspace(0.05, 1.0, 2)
else:
    val = 3000
    pred_val = 3000
    train_sizes = np.linspace(0.05, 1.0, 20)


# # Initial Model Complexity: Base Estimator Max Depth
# best max depth 6-8

# In[ ]:


if TESTING:
    parameter_range = np.arange(1, 15, 1) 
else:
    parameter_range = np.arange(1, 15, 1)
    
param_name = 'base_estimator__max_depth'
param_name_plot = 'Base Estimator Max Depth'
mnist_train_results = None
mnist_test_results = None
fashion_train_results = None
fashion_test_results = None


# In[ ]:


start_time = time.time()
for i in range(2):
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        f_name = f"{model_name}_{param_name}_MNIST"
        algorithm_name = f"{model_name} {param_name} MNIST\n Model Complexity"
        plot_title = f"{model_name} MNIST\n Model Complexity"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        f_name = f"{model_name}_{param_name}_Fashion_MNIST"
        algorithm_name = f"{model_name} {param_name} Fashion MNIST\n Model Complexity"
        plot_title = f"{model_name} Fashion MNIST\n Model Complexity"
    
    temp_train, temp_test = utl.get_model_complexity(
        classifier=AdaBoostClassifier(DecisionTreeClassifier(random_state=Random_Number)),
        train_X=temp_train_X,
        train_y=temp_train_y, parameter_name=param_name, save_dir=save_directory,
        algorithm_name=algorithm_name, parameter_range=parameter_range, cv=cv,
        n_jobs=n_jobs, verbose=5, backend='loky', param_name_for_plot=param_name_plot,
        is_NN=False, nn_range=parameter_range, plot_title=plot_title, f_name=f_name)
    if i == 0:
        mnist_train_results = temp_train
        mnist_test_results = temp_test
    else:
        fashion_train_results = temp_train
        fashion_test_results = temp_test

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# In[ ]:


utl.plot_combined_complexity("Boosting", "Base Estimator Max Depth", is_NN=False, is_SVM=False,
                             parameter_range=parameter_range, orientation='Horizontal', plt_width=12, plt_height=6,
                             mnist_train_complex=mnist_train_results, mnist_test_complex=mnist_test_results,
                             fashion_train_complex=fashion_train_results, fashion_test_complex=fashion_test_results,
                             extra_name="Max_Depth")
utl.plot_combined_complexity("Boosting", "Base Estimator Max Depth", is_NN=False, is_SVM=False,
                             parameter_range=parameter_range, orientation='Vertical', plt_height=8,
                             mnist_train_complex=mnist_train_results, mnist_test_complex=mnist_test_results,
                             fashion_train_complex=fashion_train_results, fashion_test_complex=fashion_test_results,
                             extra_name="Max_Depth")


# # Initial Model Complexity: N Estimators

# In[ ]:


if TESTING:
    parameter_range = np.arange(90, 151, 10)
else:
    parameter_range = np.arange(20, 101+1, 10)
    
param_name = 'n_estimators'
param_name_plot = 'Number of Estimators'
mnist_train_results = None
mnist_test_results = None
fashion_train_results = None
fashion_test_results = None


# In[ ]:


start_time = time.time()
for i in range(2):
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        f_name = f"{model_name}_{param_name}_MNIST"
        algorithm_name = f"{model_name} {param_name} MNIST\n Model Complexity"
        plot_title = f"{model_name} MNIST\n Model Complexity"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        f_name = f"{model_name}_{param_name}_Fashion_MNIST"
        algorithm_name = f"{model_name} {param_name} Fashion MNIST\n Model Complexity"
        plot_title = f"{model_name} Fashion MNIST\n Model Complexity"
    
    temp_train, temp_test = utl.get_model_complexity(
        classifier=AdaBoostClassifier(DecisionTreeClassifier(max_depth=6, random_state=Random_Number)),
        train_X=temp_train_X,
        train_y=temp_train_y, parameter_name=param_name, save_dir=save_directory,
        algorithm_name=algorithm_name, parameter_range=parameter_range, cv=cv,
        n_jobs=n_jobs, verbose=5, backend='loky', param_name_for_plot=param_name_plot,
        is_NN=False, nn_range=parameter_range, plot_title=plot_title, f_name=f_name)
    if i == 0:
        mnist_train_results = temp_train
        mnist_test_results = temp_test
    else:
        fashion_train_results = temp_train
        fashion_test_results = temp_test

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# In[ ]:


utl.plot_combined_complexity("Boosting", "Base Estimator Max Depth", is_NN=False, is_SVM=False,
                             parameter_range=parameter_range, orientation='Horizontal', plt_width=12, plt_height=6,
                             mnist_train_complex=mnist_train_results, mnist_test_complex=mnist_test_results,
                             fashion_train_complex=fashion_train_results, fashion_test_complex=fashion_test_results,
                             extra_name="Max_Depth_2")
utl.plot_combined_complexity("Boosting", "Base Estimator Max Depth", is_NN=False, is_SVM=False,
                             parameter_range=parameter_range, orientation='Vertical', plt_height=8,
                             mnist_train_complex=mnist_train_results, mnist_test_complex=mnist_test_results,
                             fashion_train_complex=fashion_train_results, fashion_test_complex=fashion_test_results,
                             extra_name="Max_Depth_2")


# # Initial Model Complexity: Learning Rate

# # Initial Confusion Matrix

# In[ ]:


clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=Random_Number))
clf.fit(train_X.iloc[:val, :], train_y.iloc[:val])

clf2 = AdaBoostClassifier(DecisionTreeClassifier(random_state=Random_Number))
clf2.fit(fashion_train_X.iloc[:val, :], fashion_train_y.iloc[:val])


# In[ ]:


utl.plot_combined_confusion_matrix(clf, valid_X, valid_y, clf2, fashion_valid_X, fashion_valid_y,
                                   directory=save_directory, fmt=None, plot_width=12, plot_height=6)


# # Initial Learning Curve

# In[ ]:



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
        estimator=AdaBoostClassifier(DecisionTreeClassifier(random_state=Random_Number)),
        title=title, train_X=temp_train_X,
        train_y=temp_train_y, cv=cv, f_name=f_name,
        folder="Boosting", train_sizes=train_sizes,
        save_individual=True, TESTING=True,
        n_jobs=n_jobs, backend='loky', extra_name="InitialLearningCurve")
    results.append(res)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# # Initial Metrics

# In[ ]:


clf = AdaBoostClassifier(DecisionTreeClassifier())# need to change to reflect learning curve results
p, r, f1 = utl.plot_precision_recall(clf, train_X.iloc[:val, :], train_y.iloc[:val], valid_X.iloc[:val, :],
                                 valid_y.iloc[:val], folder="Boosting", dataset_name="MNIST",
                                 plot_title="Ada Boost")


# # Gridsearch

# In[ ]:


if TESTING:
    all_parameters = {
        'n_estimators': np.arange(10, 60 + 1, 10),
        'base_estimator__max_depth': np.arange(6, 9, 1)
    }
else:
    all_parameters = {
        'n_estimators': np.arange(20, 100+1, 20),
        'base_estimator__max_depth': np.arange(6, 9, 1)
    }
    
best_mnist_estimator = None
best_fashion_estimator = None


# In[ ]:


start_time = time.time()
for i in range(2):
    if i == 0:
        temp_train_X = train_X.iloc[:2000, :]
        temp_train_y = train_y.iloc[:2000]
        algorithm_name = f"{model_name}_MNIST"
        plot_title = f"{model_name} MNIST\n Model Complexity"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        algorithm_name = f"{model_name}_Fashion_MNIST"
        plot_title = f"{model_name} Fashion MNIST\n Model Complexity"
    
    grid_results_mnist, optimized_dt_mnist = utl.run_grid_search(
        classifier=AdaBoostClassifier(DecisionTreeClassifier(random_state=Random_Number)),
        parameters=all_parameters, train_X=temp_train_X,
        train_y=temp_train_y, cv=cv,
        n_jobs=n_jobs, verbose=5, return_train_score=True,
        refit=True, save_dir=save_directory,
        algorithm_name=algorithm_name, backend='loky')
    print(f"Best Parameters:\n\t {optimized_dt_mnist.best_params_}")
    if i == 0:
        best_mnist_estimator = optimized_dt_mnist.best_estimator_
    else:
        best_fashion_estimator = optimized_dt_mnist.best_estimator_

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# # Final Learning Curve

# In[ ]:


start_time = time.time()
results = []
for i in range(2):
    print(f"Working on learning curve: {i}")
    res = {"plt": None,
           "dt_results": None,
           "cv_results": None}
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        title = f"{model_name} MNIST\n Learning Curve"
        f_name = f"{model_name}_MNIST"
        optimized_boost = best_mnist_estimator
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        title = f"{model_name} Fashion MNIST\n Learning Curve"
        f_name = f"{model_name}_Fashion_MNIST"
        optimized_boost = best_fashion_estimator
    
    res['dt_results'], res['cv_results'] = utl.plot_learning_curve(estimator=optimized_boost,
                                                                   title=title, train_X=temp_train_X,
                                                                   train_y=temp_train_y, cv=cv, f_name=f_name,
                                                                   folder="Boosting", train_sizes=train_sizes,
                                                                   save_individual=True, TESTING=True,
                                                                   n_jobs=n_jobs, backend='loky',
                                                                   extra_name="Final_Learning_Curve")
    results.append(res)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# # Final Confusion Matrix on Test set "(O_o)"

# In[ ]:


utl.plot_combined_confusion_matrix(best_mnist_estimator, test_X, test_y, best_fashion_estimator, fashion_test_X,
                                   fashion_test_y,
                                   directory=save_directory, fmt=None, plot_width=12, plot_height=6,
                                   extra_name="Boosting_Final")


# # Final Metrics

# In[ ]:


p, r, f1 = utl.plot_precision_recall(best_mnist_estimator, train_X.iloc[:val, :], train_y.iloc[:val], test_X.iloc[:val, :],
                                 test_y.iloc[:val], folder="Boosting", dataset_name="MNIST",
                                 plot_title="Ada Boost Test Set", is_final=True)


# In[ ]:


p, r, f1 = utl.plot_precision_recall(best_fashion_estimator, fashion_train_X.iloc[:val, :], fashion_train_y.iloc[:val], 
                                     fashion_test_X.iloc[:val, :], fashion_test_y.iloc[:val], 
                                     folder="Boosting", dataset_name="Fashion MNIST",
                                     plot_title="Ada Boost Test Set", is_final=True)

