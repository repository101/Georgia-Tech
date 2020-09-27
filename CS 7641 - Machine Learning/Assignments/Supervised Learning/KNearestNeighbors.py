#!/usr/bin/env python
# coding: utf-8

# # Initialize and Setup

# In[ ]:


import os
import time

import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

import ml_util_assignment_2 as utl

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500

save_directory = "figures/KNN"
model_name = "KNN"

folders = ["figures/KNN/Complexity_Analysis",
           "figures/KNN/Grid_Search_Results",
           "figures/KNN/Learning_Curves",
           "figures/KNN/Confusion_Matrix",
           "figures/KNN/Metrics"]

directories = {
    "Save Directory": "figures/KNN",
    "Initial Complexity Analysis": "figures/KNN/Initial Complexity Analysis",
    "Grid Search Results": "figures/KNN/Grid Search Results",
    "Learning Curves": "figures/KNN/Learning Curves",
    "Final Complexity Analysis": "figures/KNN/Final Complexity Analysis"
}

Random_Number = 42
TESTING = False
n_jobs = -1
cv = 5
np.random.seed(42)
# get_ipython().system('pip install pyarrow')


# In[ ]:


gathered_data = utl.setup(["MNIST"])
gathered_data_fashion = utl.setup(["Fashion-MNIST"])
train_X, train_y, valid_X, valid_y, test_X, test_y = utl.split_data(gathered_data["MNIST"]["X"],
                                                                    gathered_data["MNIST"]["y"], normalize=True,
                                                                    scale=False)
fashion_train_X, fashion_train_y, fashion_valid_X, fashion_valid_y, fashion_test_X, fashion_test_y = utl.split_data(
    gathered_data_fashion["Fashion-MNIST"]["X"],
    gathered_data_fashion["Fashion-MNIST"]["y"],
    normalize=True, scale=False)
a = fashion_train_X.min(axis=1)
b = fashion_train_X.max(axis=1)
print(f"{a.min()} {b.max()}")
import matplotlib.pyplot as plt
fashion_train_X.iloc[:1000, :].plot.hist(bins=50, alpha=0.7)
plt.savefig("Histogram.png")


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


# N Neighbors ~ default is 5, the number of neighbors used in calculations, vary this number

# Algorithm ~ 'kd_tree'       
#             'ball_tree'

if TESTING:
    val = 1000
    pred_val = 1000
    train_sizes = np.linspace(0.05, 1.00, 5)
else:
    val = 3000
    pred_val = 3000
    train_sizes = np.linspace(0.05, 1.00, 20)
print(train_sizes)


# # Initial Model Complexity

# ### Model Complexity: N Neighbors

# In[ ]:



if TESTING:
    parameter_range = np.arange(1, 3, 1)
else:
    parameter_range = np.arange(1, 15, 1)
type_1 = "kd_tree"
param_name = 'n_neighbors'
param_name_plot = 'Number of Neighbors'
mnist_train_results = None
mnist_test_results = None
fashion_train_results = None
fashion_test_results = None


# In[ ]:


start_time = time.time()
for i in range(1):
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        f_name = f"{model_name}_{type_1}_{param_name}_MNIST"
        algorithm_name = f"{model_name} {type_1} {param_name} MNIST\n Model Complexity"
        plot_title = f"{model_name} MNIST\n Model Complexity"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        f_name = f"{model_name}_{param_name}_Fashion_MNIST"
        algorithm_name = f"{model_name} {type_1} {param_name} Fashion MNIST\n Model Complexity"
        plot_title = f"{model_name} Fashion MNIST\n Model Complexity"
    
    temp_train, temp_test = utl.get_model_complexity(classifier=KNeighborsClassifier(algorithm=f"{type_1}"),
                                                     train_X=temp_train_X,
                                                     train_y=temp_train_y, parameter_name=param_name,
                                                     save_dir=save_directory,
                                                     algorithm_name=algorithm_name, parameter_range=parameter_range,
                                                     cv=cv,
                                                     n_jobs=n_jobs, verbose=5, backend='loky',
                                                     param_name_for_plot=param_name_plot,
                                                     f_name=f_name, plot_title=plot_title, extra_name='_n_neighbors_',
                                                     folder="KNN")
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


utl.plot_combined_complexity("KNN", "Number of Neighbors", is_NN=False, is_SVM=False,
                             parameter_range=parameter_range, orientation='Horizontal', plt_width=12, plt_height=6,
                             mnist_train_complex=mnist_train_results, mnist_test_complex=mnist_test_results,
                             fashion_train_complex=fashion_train_results, fashion_test_complex=fashion_test_results,
                             extra_name='_n_neighbors_', folder="KNN")
utl.plot_combined_complexity("KNN", "Number of Neighbors", is_NN=False, is_SVM=False,
                             parameter_range=parameter_range, orientation='Vertical', plt_height=8,
                             mnist_train_complex=mnist_train_results, mnist_test_complex=mnist_test_results,
                             fashion_train_complex=fashion_train_results, fashion_test_complex=fashion_test_results,
                             extra_name='_n_neighbors_', folder="KNN")


# ### Model Complexity: Leaf Size

# In[ ]:


if TESTING:
    parameter_range = np.arange(1, 3, 1)
else:
    parameter_range = np.arange(1, 15, 1)
type_1 = "kd_tree"
param_name = 'leaf_size'
param_name_plot = 'Leaf Size'
mnist_train_results_leaf = None
mnist_test_results_leaf = None
fashion_train_results_leaf = None
fashion_test_results_leaf = None


# In[ ]:


start_time = time.time()
for i in range(2):
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        f_name = f"{model_name}_{type_1}_{param_name}_MNIST"
        algorithm_name = f"{model_name} {type_1} {param_name} MNIST\n Model Complexity"
        plot_title = f"{model_name} MNIST\n Model Complexity"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        f_name = f"{model_name}_{param_name}_Fashion_MNIST"
        algorithm_name = f"{model_name} {type_1} {param_name} Fashion MNIST\n Model Complexity"
        plot_title = f"{model_name} Fashion MNIST\n Model Complexity"
    
    temp_train, temp_test = utl.get_model_complexity(classifier=KNeighborsClassifier(algorithm=f"{type_1}"),
                                                     train_X=temp_train_X,
                                                     train_y=temp_train_y, parameter_name=param_name,
                                                     save_dir=save_directory,
                                                     algorithm_name=algorithm_name, parameter_range=parameter_range,
                                                     cv=cv,
                                                     n_jobs=n_jobs, verbose=5, backend='loky',
                                                     param_name_for_plot=param_name_plot,
                                                     f_name=f_name, plot_title=plot_title, extra_name='_leaf_size_',
                                                     folder="KNN")
    if i == 0:
        mnist_train_results_leaf = temp_train
        mnist_test_results_leaf = temp_test
    else:
        fashion_train_results_leaf = temp_train
        fashion_test_results_leaf = temp_test

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# In[ ]:


utl.plot_combined_complexity("KNN", "Leaf Size", is_NN=False, is_SVM=False,
                             parameter_range=parameter_range, orientation='Horizontal', plt_width=12, plt_height=6,
                             mnist_train_complex=mnist_train_results_leaf, mnist_test_complex=mnist_test_results_leaf,
                             fashion_train_complex=fashion_train_results_leaf,
                             fashion_test_complex=fashion_test_results_leaf,
                             extra_name='_leaf_size_', folder="KNN")
utl.plot_combined_complexity("KNN", "Leaf Size", is_NN=False, is_SVM=False,
                             parameter_range=parameter_range, orientation='Vertical', plt_height=8,
                             mnist_train_complex=mnist_train_results_leaf, mnist_test_complex=mnist_test_results_leaf,
                             fashion_train_complex=fashion_train_results_leaf,
                             fashion_test_complex=fashion_test_results_leaf,
                             extra_name='_leaf_size_', folder="KNN")


# # Initial Confusion Matrix

# In[ ]:


clf = KNeighborsClassifier(algorithm=f"{type_1}")
clf.fit(train_X.iloc[:val, :], train_y.iloc[:val])

clf2 = KNeighborsClassifier(algorithm=f"{type_1}")
clf2.fit(fashion_train_X.iloc[:val, :], fashion_train_y.iloc[:val])


# In[ ]:


utl.plot_combined_confusion_matrix(clf, valid_X, valid_y, clf2, fashion_valid_X, fashion_valid_y,
                                   directory=save_directory, fmt="d", plot_width=12, plot_height=6,
                                   extra_name="Initial_Confusion", folder="KNN")


# # Initial Learning Curve

# In[ ]:


knn = KNeighborsClassifier(algorithm="ball_tree")


# In[ ]:


start_time = time.time()
results = []
for i in range(2):
    print(f"Working on learning curve: {i}")
    res = {'plt': None,
           'dt_results': None,
           'cv_results': None}
    
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
    
    res['knn_results'], res['cv_results'] = utl.plot_learning_curve(estimator=knn, title=title,
                                                                    train_X=temp_train_X, train_y=temp_train_y, cv=cv,
                                                                    f_name=f_name, train_sizes=train_sizes,
                                                                    save_individual=True,
                                                                    TESTING=True, backend='loky', n_jobs=n_jobs,
                                                                    folder="KNN")
    results.append(res)

end_time = time.time()
elapsed_time = end_time - start_time


# # Initial Metrics

# In[ ]:


clf = KNeighborsClassifier(algorithm=f"{type_1}") # need to change to reflect learning curve results
p, r, f1 = utl.plot_precision_recall(clf, train_X.iloc[:val, :], train_y.iloc[:val], valid_X.iloc[:val, :],
                                 valid_y.iloc[:val], folder="KNN", dataset_name="MNIST",
                                 plot_title="K Nearest Neighbors")


# # Gridsearch

# In[ ]:


if TESTING:
    all_parameters = {
        'n_neighbors': np.arange(5, 11, 5),
        'leaf_size': np.arange(10, 11, 10),
    }
else:
    all_parameters = {
        'n_neighbors': np.arange(1, 5+1, 5),
        'weights': ['uniform', 'distance'],
        'leaf_size': np.arange(1, 50+1, 5),
        'algorithm': ['ball_tree', 'kd_tree']
    }


best_mnist_estimator = None
best_fashion_estimator = None


# In[ ]:


start_time = time.time()
for i in range(2):
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        f_name = f"{model_name}_MNIST"
        algorithm_name = f"{model_name} MNIST Model Complexity"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        f_name = f"{model_name}_Fashion_MNIST"
        algorithm_name = f"{model_name} Fashion MNIST Model Complexity"
    
    grid_results_mnist, optimized_dt_mnist = utl.run_grid_search(classifier=KNeighborsClassifier(),
                                                                 parameters=all_parameters, train_X=temp_train_X,
                                                                 train_y=temp_train_y, cv=cv,
                                                                 n_jobs=n_jobs, verbose=10, return_train_score=True,
                                                                 refit=True, save_dir=save_directory,
                                                                 algorithm_name=algorithm_name, backend='loky',
                                                                 folder="KNN")
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
    res = {'plt': None,
           'dt_results': None,
           'cv_results': None}
    
    if i == 0:
        temp_train_X = test_X.iloc[:val, :]
        temp_train_y = test_y.iloc[:val]
        title = f"{model_name} MNIST\n Final Learning Curve"
        f_name = f"{model_name}_MNIST"
        optimized_knn = best_mnist_estimator
    else:
        temp_train_X = fashion_test_X.iloc[:val, :]
        temp_train_y = fashion_test_y.iloc[:val]
        title = f"{model_name} Fashion MNIST\n Final Learning Curve"
        f_name = f"{model_name}_Fashion_MNIST"
        optimized_knn = best_fashion_estimator
    
    res['knn_results'], res['cv_results'] = utl.plot_learning_curve(estimator=optimized_knn, title=title,
                                                                    train_X=temp_train_X,
                                                                    train_y=temp_train_y, cv=cv, f_name=f_name,
                                                                    train_sizes=train_sizes,
                                                                    save_individual=True, TESTING=True, backend='loky',
                                                                    n_jobs=n_jobs, folder="KNN")
    results.append(res)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# # Final Confusion Matrix on Test set "(O_o)"

# In[ ]:


utl.plot_combined_confusion_matrix(best_mnist_estimator, test_X, test_y, best_fashion_estimator, fashion_test_X,
                                   fashion_test_y,
                                   directory=save_directory, fmt="d", plot_width=12, plot_height=6, folder="KNN")


# # Final Metrics

# In[ ]:


p, r, f1 = utl.plot_precision_recall(best_mnist_estimator, train_X.iloc[:val, :], train_y.iloc[:val], test_X.iloc[:val, :],
                                 test_y.iloc[:val], folder="KNN", dataset_name="MNIST",
                                 plot_title="K Nearest Neighbors Test Set", is_final=True)


# In[ ]:


p, r, f1 = utl.plot_precision_recall(best_fashion_estimator, fashion_train_X.iloc[:val, :], 
                                     fashion_train_y.iloc[:val], fashion_test_X.iloc[:val, :],
                                 fashion_test_y.iloc[:val], folder="KNN", dataset_name="Fashion MNIST",
                                 plot_title="K Nearest Neighbors Test Set", is_final=True)

