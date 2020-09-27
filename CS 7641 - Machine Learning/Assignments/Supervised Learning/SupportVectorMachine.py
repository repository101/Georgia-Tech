#!/usr/bin/env python
# coding: utf-8

# # Initialize and Setup

# In[ ]:


import os
import time

import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve

import ml_util_assignment_2 as utl

save_directory = "figures/SVM"
model_name = "SVM"

folders = ["figures/SVM/Grid_Search_Results",
           "figures/SVM/Complexity_Analysis",
           "figures/SVM/Learning_Curves",
           "figures/SVM/Confusion_Matrix",
           "figures/SVM/Metrics"]

directories = {
    "Save Directory": "figures/SVM",
    "Initial Complexity Analysis": "figures/SVM/Initial Complexity Analysis",
    "Grid Search Results": "figures/SVM/Grid Search Results",
    "Learning Curves": "figures/SVM/Learning Curves",
    "Final Complexity Analysis": "figures/SVM/Final Complexity Analysis"
}

Random_Number = 42
TESTING =  False
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


# Kernel ~ 'rbf'
#          'sigmoid'

# degree ~ The degree of the polynomial (only used with poly kernel)

if TESTING:
    val = 600
    pred_val = 600
    train_sizes = np.linspace(0.05, 1.0, 5)
else:
    val = 3000
    pred_val = 3000
    train_sizes = np.linspace(0.05, 1.0, 3)


# In[ ]:


a = 10. ** np.arange(-4, 4, 1)
if TESTING:
    e = np.sort(a)
else:
    e = np.sort(np.hstack((a, a * 3, a * 5, a * 7)))


# In[ ]:


# Vary the kernel { "rbf", "sigmoid"}

# Vary the training sizes


# In[ ]:


e


# # Initial Model Complexity: RBF

# In[ ]:


parameter_range = e
param_name = "C"
param_name_plot = "C"
kernel_name = "RBF"


# In[ ]:


mnist_train_results_rbf = None
mnist_test_results_rbf = None
fashion_train_results_rbf = None
fashion_test_results_rbf = None


# In[ ]:


start_time = time.time()
for i in range(2):
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        f_name = f"{model_name}_{param_name}_MNIST_{kernel_name}"
        algorithm_name = f"{model_name} {param_name} MNIST\n Model Complexity {kernel_name}"
        plot_title = f"{model_name} MNIST\n Model Complexity Kernel: {kernel_name}"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        f_name = f"{model_name}_{param_name}_Fashion_MNIST_{kernel_name}"
        algorithm_name = f"{model_name} {param_name} Fashion MNIST\n Model Complexity {kernel_name}"
        plot_title = f"{model_name} Fashion MNIST\n Model Complexity Kernel: {kernel_name}"
    
    temp_train, temp_test = utl.get_model_complexity(classifier=SVC(kernel=kernel_name.lower(), cache_size=400),
                                                     train_X=temp_train_X, train_y=temp_train_y, parameter_name=param_name,
                                                     save_dir=save_directory, algorithm_name=algorithm_name, 
                                                     parameter_range=parameter_range, cv=5,
                                                     n_jobs=n_jobs, verbose=5, backend='loky',
                                                     param_name_for_plot=param_name_plot,
                                                     f_name=f_name, plot_title=plot_title, is_SVM=True,
                                                     extra_name="RBF", folder="SVM", use_log_x=True)
    if i == 0:
        mnist_train_results_rbf = temp_train
        mnist_test_results_rbf = temp_test
    else:
        fashion_train_results_rbf = temp_train
        fashion_test_results_rbf = temp_test

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# In[ ]:


utl.plot_combined_complexity("SVM", "C", is_NN=False, is_SVM=True, parameter_range=parameter_range,
                             orientation='Horizontal', plt_width=12, plt_height=6, extra_name="RBF",
                             mnist_train_complex=mnist_train_results_rbf, mnist_test_complex=mnist_test_results_rbf,
                             fashion_train_complex=fashion_train_results_rbf, use_log_x=True, use_saved=False,
                             fashion_test_complex=fashion_test_results_rbf, folder="SVM", only_one=True, which_one='rbf')


# # Initial Confusion Matrix: RBF

# In[ ]:


clf = SVC(kernel='rbf', cache_size=400)
clf.fit(train_X.iloc[:val, :], train_y.iloc[:val])

clf2 = SVC(kernel='rbf', cache_size=400)
clf2.fit(fashion_train_X.iloc[:val, :], fashion_train_y.iloc[:val])


# In[ ]:


utl.plot_combined_confusion_matrix(clf, valid_X, valid_y, clf2, fashion_valid_X, fashion_valid_y,
                                   directory=save_directory, fmt=None, plot_width=12, plot_height=6, extra_name="RBF",
                                   folder="SVM")


# # Initial Learning Curve: RBF

# In[ ]:


initial_svm_rbf = SVC(kernel='rbf')


# In[ ]:


start_time = time.time()
results = []
for i in range(2):
    print(f"Working on learning curve: {i}")
    res = {"plt": None,
           "svm_results": None,
           "cv_results": None}
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        title = f"{model_name} MNIST Learning Curve"
        f_name = f"{model_name}_MNIST"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        title = f"{model_name} Fashion MNIST Learning Curve"
        f_name = f"{model_name}_Fashion_MNIST"
    
    res['svm_results'], res['cv_results'] = utl.plot_learning_curve(estimator=initial_svm_rbf, title=title,
                                                                    train_X=temp_train_X, train_y=temp_train_y, cv=cv,
                                                                    f_name=f_name, train_sizes=train_sizes,
                                                                    folder="SVM",
                                                                    save_individual=True, TESTING=True, backend='loky',
                                                                    n_jobs=n_jobs,
                                                                    extra_name="RBF")
    results.append(res)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# # Initial Metrics: RBF

# In[ ]:


clf = SVC(kernel='rbf', cache_size=400) # need to change to reflect learning curve results
p, r, f1 = utl.plot_precision_recall(clf, train_X.iloc[:2000, :], train_y.iloc[:2000], valid_X.iloc[:2000, :],
                                 valid_y.iloc[:2000], folder="SVM", dataset_name="MNIST",
                                 plot_title="Support Vector Machine")


# # Initial Model Complexity: Linear

# In[ ]:


parameter_range = np.sort(e.flatten())
param_name = "C"
param_name_plot = "C"
kernel_name = "Linear"


# In[ ]:


mnist_train_results_lin = None
mnist_test_results_lin = None
fashion_train_results_lin = None
fashion_test_results_lin = None


# In[ ]:


start_time = time.time()
for i in range(2):
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        f_name = f"{model_name}_{param_name}_MNIST_{kernel_name}"
        algorithm_name = f"{model_name} {param_name} MNIST\n Model Complexity {kernel_name}"
        plot_title = f"{model_name} MNIST\n Model Complexity Kernel: {kernel_name}"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        f_name = f"{model_name}_{param_name}_Fashion_MNIST_{kernel_name}"
        algorithm_name = f"{model_name} {param_name} Fashion MNIST\n Model Complexity {kernel_name}"
        plot_title = f"{model_name} Fashion MNIST\n Model Complexity Kernel: {kernel_name}"
    
    temp_train, temp_test = utl.get_model_complexity(classifier=SVC(kernel=kernel_name.lower(), cache_size=400),
                                                     train_X=temp_train_X,
                                                     train_y=temp_train_y, parameter_name=param_name,
                                                     save_dir=save_directory,
                                                     algorithm_name=algorithm_name, parameter_range=parameter_range,
                                                     cv=5, use_log_x=True,
                                                     n_jobs=n_jobs, verbose=5, backend='loky',
                                                     param_name_for_plot=param_name_plot,
                                                     f_name=f_name, plot_title=plot_title, is_SVM=True,
                                                     extra_name="Linear", folder="SVM")
    if i == 0:
        mnist_train_results_lin = temp_train
        mnist_test_results_lin = temp_test
    else:
        fashion_train_results_lin = temp_train
        fashion_test_results_lin = temp_test

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# In[ ]:


utl.plot_combined_complexity("SVM", "C", is_NN=False, is_SVM=True, parameter_range=parameter_range,
                             orientation='Horizontal', plt_width=12, plt_height=6, extra_name="Linear",
                             mnist_train_complex=mnist_train_results_lin, mnist_test_complex=mnist_test_results_lin,
                             fashion_train_complex=fashion_train_results_lin, use_log_x=True, use_saved=False,
                             fashion_test_complex=fashion_test_results_lin, folder="SVM", only_one=True, which_one='linear',
                             ylim=(0.1, 1.05))


# # Initial Confusion Matrix: Linear

# In[ ]:


clf = SVC(kernel='linear', cache_size=400)
clf.fit(train_X.iloc[:val, :], train_y.iloc[:val])

clf2 = SVC(kernel='linear', cache_size=400)
clf2.fit(fashion_train_X.iloc[:val, :], fashion_train_y.iloc[:val])


# In[ ]:


utl.plot_combined_confusion_matrix(clf, valid_X, valid_y, clf2, fashion_valid_X, fashion_valid_y,
                                   directory=save_directory, fmt=None, plot_width=12, plot_height=6,
                                   extra_name="Linear", folder="SVM")


# # Initial Learning Curve: Linear

# In[ ]:


initial_svm_linear = SVC(kernel='linear')


# In[ ]:


start_time = time.time()
results = []
for i in range(2):
    print(f"Working on learning curve: {i}")
    res = {"plt": None,
           "svm_results": None,
           "cv_results": None}
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        title = f"{model_name} MNIST Learning Curve"
        f_name = f"{model_name}_MNIST"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        title = f"{model_name} Fashion MNIST Learning Curve"
        f_name = f"{model_name}_Fashion_MNIST"
    
    res['svm_results'], res['cv_results'] = utl.plot_learning_curve(estimator=initial_svm_linear, title=title,
                                                                    train_X=temp_train_X, train_y=temp_train_y, cv=cv,
                                                                    f_name=f_name, train_sizes=train_sizes,
                                                                    folder="SVM",
                                                                    save_individual=True, TESTING=True, backend='loky',
                                                                    n_jobs=n_jobs,
                                                                    extra_name="Linear")
    results.append(res)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# In[ ]:


print("hey")


# # Initial Metrics: Linear

# In[ ]:


clf = SVC(kernel='linear', cache_size=400) # need to change to reflect learning curve results
p, r, f1 = utl.plot_precision_recall(clf, train_X.iloc[:val, :], train_y.iloc[:val], valid_X.iloc[:val, :],
                                 valid_y.iloc[:val], folder="SVM", dataset_name="MNIST",
                                 plot_title="Support Vector Machine")


# # Gridsearch: RBF

# In[ ]:


rbf_a = 10. ** np.arange(-7, -3, 1)

if TESTING:
    gamma_rbf = np.sort(rbf_a)

else:
    gamma_rbf = np.sort(np.hstack((rbf_a, rbf_a * 3, rbf_a * 5, rbf_a * 7)))


params_rbf = {'kernel': ['rbf'], 'C': 10. ** np.arange(-2, 5, 1), 'gamma': gamma_rbf}

params_lin = {'kernel': ['linear'], 'C': 10. ** np.arange(-4, 1, 1), 'gamma': ['scale'], 'class_weight': ['balanced']}
parameters_rbf = ParameterGrid(params_rbf)
parameters_lin = ParameterGrid(params_lin)


# In[ ]:


best_rbf_mnist_estimator = None
best_rbf_fashion_estimator = None
res = []
start_time = time.time()
for i in range(2):
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        algorithm_name = f"{model_name}_MNIST"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        algorithm_name = f"{model_name}_Fashion_MNIST"
    
    grid_results_mnist_rbf, optimized_svm_mnist_rbf = utl.run_grid_search(
        classifier=SVC(cache_size=400, random_state=Random_Number, kernel="rbf"), parameters=params_rbf,
        train_X=temp_train_X, train_y=temp_train_y, cv=cv,
        n_jobs=n_jobs, verbose=5, return_train_score=True, refit=True,
        save_dir=save_directory, algorithm_name=algorithm_name,
        backend='loky', extra_f_name="RBF", folder="SVM")
    if i == 0:
        best_rbf_mnist_estimator = optimized_svm_mnist_rbf.best_estimator_
    else:
        best_rbf_fashion_estimator = optimized_svm_mnist_rbf.best_estimator_
    
    print(f"\nBest Parameters:\n\t{optimized_svm_mnist_rbf.best_params_}")
    res.append((grid_results_mnist_rbf, optimized_svm_mnist_rbf))
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# # Gridsearch: Linear

# In[ ]:


best_lin_mnist_estimator = None
best_lin_fashion_estimator = None
res2 = []
start_time = time.time()
for i in range(2):
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        algorithm_name = f"{model_name}_MNIST"
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        algorithm_name = f"{model_name}_Fashion_MNIST"
    
    grid_results_mnist_lin, optimized_svm_mnist_lin = utl.run_grid_search(
        classifier=SVC(cache_size=400, random_state=Random_Number), parameters=params_lin,
        train_X=temp_train_X, train_y=temp_train_y, cv=cv,
        n_jobs=n_jobs, verbose=5, return_train_score=True, refit=True,
        save_dir=save_directory, algorithm_name=algorithm_name,
        backend='loky', extra_f_name="Linear", folder="SVM")
    if i == 0:
        best_lin_mnist_estimator = optimized_svm_mnist_lin.best_estimator_
    else:
        best_lin_fashion_estimator = optimized_svm_mnist_lin.best_estimator_
    
    print(f"\nBest Parameters:\n\t{optimized_svm_mnist_lin.best_params_}")
    print(f"\nBest Parameters:\n\t{optimized_svm_mnist_lin.best_score_}")
    res2.append((grid_results_mnist_lin, optimized_svm_mnist_lin))
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# # Final Learning Curve: RBF

# In[ ]:


start_time = time.time()
results = []
for i in range(2):
    print(f"Working on learning curve: {i}")
    res = {"plt": None,
           "svm_results": None,
           "cv_results": None}
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        title = f"{model_name} MNIST Learning Curve"
        f_name = f"{model_name}_MNIST"
        optimized_svm = best_rbf_mnist_estimator
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        title = f"{model_name} Fashion MNIST Learning Curve"
        f_name = f"{model_name}_Fashion_MNIST"
        optimized_svm = best_rbf_fashion_estimator
    
    res['svm_results'], res['cv_results'] = utl.plot_learning_curve(estimator=optimized_svm, title=title,
                                                                    train_X=temp_train_X, train_y=temp_train_y, cv=cv,
                                                                    f_name=f_name, train_sizes=train_sizes,
                                                                    n_jobs=n_jobs, folder="SVM",
                                                                    save_individual=True, TESTING=True, backend='loky',
                                                                    extra_name="RBF_Final")
    results.append(res)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# # Final Learning Curve: Linear

# In[ ]:


start_time = time.time()
results = []
for i in range(2):
    print(f"Working on learning curve: {i}")
    res = {"plt": None,
           "svm_results": None,
           "cv_results": None}
    if i == 0:
        temp_train_X = train_X.iloc[:val, :]
        temp_train_y = train_y.iloc[:val]
        title=f"{model_name} MNIST Learning Curve"
        f_name= f"{model_name}_MNIST"
        optimized_svm_linear = best_lin_mnist_estimator
    else:
        temp_train_X = fashion_train_X.iloc[:val, :]
        temp_train_y = fashion_train_y.iloc[:val]
        title=f"{model_name} Fashion MNIST Learning Curve"
        f_name= f"{model_name}_Fashion_MNIST"
        optimized_svm_linear = best_lin_fashion_estimator
    
    res['svm_results'], res['cv_results'] = utl.plot_learning_curve(estimator=optimized_svm_linear, title=title, 
                                                       train_X=temp_train_X, train_y=temp_train_y, cv=cv, 
                                                       f_name=f_name, train_sizes=train_sizes, folder="SVM",
                                                       save_individual=True, TESTING=True, backend='loky', 
                                                                                n_jobs=n_jobs, extra_name="Linear_Final")
    results.append(res)
    
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# # Final Confusion Matrix on Test set "(O_o)" - RBF

# In[ ]:


utl.plot_combined_confusion_matrix(best_rbf_mnist_estimator, test_X, test_y, best_rbf_fashion_estimator, fashion_test_X,
                                   fashion_test_y, directory=save_directory, fmt=None, plot_width=12, plot_height=6,
                                   extra_name="RBF_Final", folder="SVM")


# # Final Confusion Matrix on Test set "(O_o)" - Linear

# In[ ]:


utl.plot_combined_confusion_matrix(best_lin_mnist_estimator, test_X, test_y, best_lin_fashion_estimator, fashion_test_X,
                                   fashion_test_y, directory=save_directory, fmt=None, plot_width=12, plot_height=6,
                                   extra_name="Linear_Final", folder="SVM")


# # Final Metrics - RBF

# In[ ]:


p, r, f1 = utl.plot_precision_recall(best_rbf_mnist_estimator, train_X.iloc[:val, :], train_y.iloc[:val],
                                     test_X.iloc[:val, :], test_y.iloc[:val], folder="SVM", dataset_name="MNIST",
                                     plot_title="Support Vector Machine Test Set", is_final=True)


# In[ ]:


p, r, f1 = utl.plot_precision_recall(best_rbf_mnist_estimator, fashion_train_X.iloc[:val, :], 
                                     fashion_train_y.iloc[:val], fashion_test_X.iloc[:val, :],
                                     fashion_test_y.iloc[:val], folder="SVM", dataset_name="Fashion MNIST",
                                     plot_title="Support Vector Machine Test Set", is_final=True)


# # Final Metrics - Linear

# In[ ]:


p, r, f1 = utl.plot_precision_recall(best_lin_mnist_estimator, train_X.iloc[:val, :], train_y.iloc[:val],
                                     test_X.iloc[:val, :],test_y.iloc[:val], folder="SVM", dataset_name="MNIST",
                                     plot_title="Support Vector Machine Test Set", is_final=True)


# In[ ]:


p, r, f1 = utl.plot_precision_recall(best_lin_mnist_estimator, fashion_train_X.iloc[:val, :], 
                                     fashion_train_y.iloc[:val], fashion_test_X.iloc[:val, :],
                                     fashion_test_y.iloc[:val], folder="SVM", dataset_name="Fashion MNIST",
                                     plot_title="Support Vector Machine Test Set", is_final=True)

