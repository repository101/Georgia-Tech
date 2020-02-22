#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from LoadData import LoadData
from sklearn.utils import parallel_backend
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import classification_report, confusion_matrix


TESTING = True
DECISION_TREE = False
SUPPORT_VECTOR = True
NEURAL_NET = False
K_NEAREST = False
BOOSTING = False
NORMALIZE_DATA = False
USE_PCA = True
DataSetName = "MNIST"
dataset = "MNIST"


# In[2]:


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    with plt.style.context('ggplot'):
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ =             learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt


# In[ ]:


"""

1. Call train_test_split to get the training data and test data (x_train, x_test, y_train, and y_test respectively.
2. Save the test data (x_test, y_test) until the end.
3. Repeat the following for each algorithm: SVM, Decision Tree, Boosting, NN, and KNN:
4. Select a set of hyperparameters to optimize.
5. Pass these and the classifier object (DecisionTree for example) into GridSearchCV.
6. This returns a clf object that you then call clf.fit(x_train, y_train) to get the best hyperparameters.
7. Get the Learning Curve by taking the clf.best_estimator_ and passing it into learning_curve function of scikit learn.
8. Plot the learning curve results
9. Get the Model Complexity data by taking the clf.best_estimator_ and passing it into the validation_curve function. Pass in one hyperparameter to tune that was not tuned in your GridSearchCV process in number 5 above.
10. Plot the results of validation_curve.
11. Run the x_test and y_test through clf.best_estimator_ predict to see how the model performs.

"""


# In[3]:


"""
STEP 1.

"""
classifiers = ["SVM", "Decision Tree", "Neural Network", "K-Nearest Neighbors", "Boosting"]
n_jobs = -1
cross_validation = 5
verbose_int = 3

cwd = pathlib.Path().absolute()

if DataSetName == "MNIST":
    training_data_path = "{}/sm-mnist-train-data.csv".format(cwd)
    testing_data_path = "{}/sm-mnist-test-data.csv".format(cwd)
else:
    training_data_path = "{}/sm-fashion-mnist-train-data.csv".format(cwd)
    testing_data_path = "{}/sm-fashion-mnist-test-data.csv".format(cwd)

training_labels, training_data, _ = LoadData(training_data_path)
testing_labels, testing_data, _ = LoadData(testing_data_path)

Scaler = StandardScaler().fit(training_data)

training_data = Scaler.transform(training_data)
testing_data = Scaler.transform(testing_data)


# In[4]:


"""
STEP 3.

"""

svm_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
             'gamma': [1e-4, 1e-3, 1e-2, 1e-1],
             'max_iter': [1000]}

decision_tree_params = {'criterion': ['gini', 'entropy'],
                       'max_depth': [50, 100, 150, 200],
                       'min_samples_split': [i for i in range(4, 13, 4)]}

neural_network_params = {'activation': ['identity', 'relu'],
                        'solver': ['lbfgs', 'sgd', 'adam'],
                        'max_iter': [1000],
                        'alpha': [1e-4, 1e-3, 1e-2, 1e-1]}

knn_params = {'n_neighbors': [3, 5, 7, 9, 11],
             'algorithm': ['ball_tree', 'kd_tree', 'brute']}

boosting_params = {'n_estimators': [50, 100, 150],
                  'max_depth': [1, 2, 3],
                  'learning_rate': [1e-1, 1e-2, 1e-3]}


# In[5]:


X = training_data[:2000]
Validation_X = X[:400]
y = training_labels[:2000]
validation_y = y[:400]
container = {"train_sizes": None, "train_scores": None, "test_scores": None, "fit_times": None}
train_sizes = np.linspace(.1, 1.0, 50)
colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]


# In[15]:


svm_segment = "Support Vector Machine"
svm_estimator=svm.SVC()
svm_grid_search = GridSearchCV(svm_estimator, param_grid=svm_params, n_jobs=n_jobs, verbose=verbose_int)
svm_learning_curve_results = {"train_sizes": None, "train_scores": None, "test_scores": None, "fit_times": None}
svm_validation_curve_results = {"train_scores": None, "test_scores": None}
svm_param_name = "C"


# In[31]:


with parallel_backend('threading'):
    svm_grid_search.fit(X,y)


# In[32]:


with parallel_backend('threading'):
    svm_learning_curve_results["train_sizes"], svm_learning_curve_results["train_scores"], svm_learning_curve_results["test_scores"], svm_learning_curve_results["fit_times"], _ = learning_curve(svm_grid_search.best_estimator_, X, y, cv=cross_validation, verbose=3, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)


# In[33]:


with parallel_backend('threading'):    
    svm_validation_curve_results["train_scores"], svm_validation_curve_results["test_scores"] = validation_curve(svm_grid_search.best_estimator_, X=Validation_X, y=validation_y, n_jobs=n_jobs, cv=cross_validation, verbose=3, param_name=svm_param_name, param_range=np.linspace(0.1, 2.5, num=50))

plt.close("all")


# In[54]:



with plt.style.context('ggplot'):
    svm_train_scores_mean = np.mean(svm_learning_curve_results["train_scores"], axis=1)
    svm_train_scores_std = np.std(svm_learning_curve_results["train_scores"], axis=1)
    svm_test_scores_mean = np.mean(svm_learning_curve_results["test_scores"], axis=1)
    svm_test_scores_std = np.std(svm_learning_curve_results["test_scores"], axis=1)
    
    fig0, ax0 = plt.subplots()
    ax0.set_xlabel("Percent of Training Set")
    ax0.set_ylabel("Accuracy (%)", color='tab:orange')
    ax0.set_title("{} {} Learning Curve".format(svm_segment, dataset))
    ax0.tick_params(axis='y', labelcolor="black")
    ax0.set_ylim(0, 1.1)
    lw = 2
    plt.plot([i for i in range(50)], svm_train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between([i for i in range(50)], svm_train_scores_mean - svm_train_scores_std,
                     svm_train_scores_mean + svm_train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot([i for i in range(50)], svm_test_scores_mean, label="Testing score",
                 color="navy", lw=lw)
    plt.fill_between([i for i in range(50)], svm_test_scores_mean - svm_test_scores_std,
                     svm_test_scores_mean + svm_test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig("{}_{}_Learning_Curve.png".format(svm_segment, dataset))
    plt.show()
    
plt.close("all")
    


# In[55]:


with plt.style.context('ggplot'):
    svm_train_scores_mean = np.mean(svm_validation_curve_results["train_scores"], axis=1)
    svm_train_scores_std = np.std(svm_validation_curve_results["train_scores"], axis=1)
    svm_test_scores_mean = np.mean(svm_validation_curve_results["test_scores"], axis=1)
    svm_test_scores_std = np.std(svm_validation_curve_results["test_scores"], axis=1)

    plt.title("Validation Curve with {} \n {}".format(svm_segment, dataset))
    plt.xlabel("{}".format(svm_param_name))
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot([i for i in range(50)], svm_train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between([i for i in range(50)], svm_train_scores_mean - svm_train_scores_std,
                     svm_train_scores_mean + svm_train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot([i for i in range(50)], svm_test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between([i for i in range(50)], svm_test_scores_mean - svm_test_scores_std,
                     svm_test_scores_mean + svm_test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig("{}_{}_Validation_Curve.png".format(svm_segment, dataset))
    plt.show()
plt.close('all')


# In[12]:


dt_segment = "Decision Tree"
dt_estimator = DecisionTreeClassifier()
dt_grid_search = GridSearchCV(dt_estimator, param_grid=decision_tree_params,  n_jobs=n_jobs, verbose=verbose_int)
dt_learning_curve_results = {"train_sizes": None, "train_scores": None, "test_scores": None, "fit_times": None}
dt_validation_curve_results = {"train_scores": None, "test_scores": None}


# In[13]:


with parallel_backend('threading'):
    dt_grid_search.fit(X,y)
  


# In[16]:


with parallel_backend('threading'):    
    dt_learning_curve_results["train_sizes"], dt_learning_curve_results["train_scores"], dt_learning_curve_results["test_scores"], dt_learning_curve_results["fit_times"], _ = learning_curve(dt_grid_search.best_estimator_, X, y, cv=cross_validation, n_jobs=n_jobs, verbose=3, train_sizes=train_sizes, return_times=True)
 


# In[17]:


with parallel_backend('threading'):    
    dt_validation_curve_results["train_scores"], dt_validation_curve_results["test_scores"] = validation_curve(dt_grid_search.best_estimator_, X=Validation_X, y=validation_y, n_jobs=n_jobs, cv=cross_validation, verbose=3, param_name='min_samples_leaf', param_range=[i for i in range(1, 51)])

plt.close("all")


# In[18]:



with plt.style.context('ggplot'):
    dt_train_scores_mean = np.mean(dt_learning_curve_results["train_scores"], axis=1)
    dt_train_scores_std = np.std(dt_learning_curve_results["train_scores"], axis=1)
    dt_test_scores_mean = np.mean(dt_learning_curve_results["test_scores"], axis=1)
    dt_test_scores_std = np.std(dt_learning_curve_results["test_scores"], axis=1)
    
    fig0, ax0 = plt.subplots()
    ax0.set_xlabel("Percent of Training Set")
    ax0.set_ylabel("Accuracy (%)", color='tab:orange')
    ax0.set_title("Decision_Tree Learning Curve")
    ax0.tick_params(axis='y', labelcolor="black")
    ax0.set_ylim(0, 1.1)

    lw = 2
    plt.plot([i for i in range(50)], dt_train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    
    plt.fill_between([i for i in range(50)], dt_train_scores_mean - dt_train_scores_std,
                     dt_train_scores_mean + dt_train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot([i for i in range(50)], dt_test_scores_mean, label="Testing score",
                 color="navy", lw=lw)
    plt.fill_between([i for i in range(50)], dt_test_scores_mean - dt_test_scores_std,
                     dt_test_scores_mean + dt_test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig("{}_{}_Learning_Curve.png".format(dt_segment, dataset))
    plt.show()
    
plt.close("all")


# In[19]:


with plt.style.context('ggplot'):
    dt_train_scores_mean = np.mean(dt_validation_curve_results["train_scores"], axis=1)
    dt_train_scores_std = np.std(dt_validation_curve_results["train_scores"], axis=1)
    dt_test_scores_mean = np.mean(dt_validation_curve_results["test_scores"], axis=1)
    dt_test_scores_std = np.std(dt_validation_curve_results["test_scores"], axis=1)

    plt.title("Validation Curve with {} \n {}".format(dt_segment, dataset))
    plt.xlabel("{}".format('min_samples_leaf'))
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot([i for i in range(50)], dt_train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between([i for i in range(50)], dt_train_scores_mean - dt_train_scores_std,
                     dt_train_scores_mean + dt_train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot([i for i in range(50)], dt_test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between([i for i in range(50)], dt_test_scores_mean - dt_test_scores_std,
                     dt_test_scores_mean + dt_test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig("{}_{}_Validation_Curve.png".format(dt_segment, dataset))
    plt.show()


# In[11]:


nn_segment = "Neural Network"
nn_estimator = MLPClassifier()
nn_grid_search = GridSearchCV(nn_estimator, param_grid=neural_network_params, n_jobs=n_jobs, verbose=verbose_int)
nn_learning_curve_results = {"train_sizes": None, "train_scores": None, "test_scores": None, "fit_times": None}
nn_validation_curve_results = {"train_scores": None, "test_scores": None}


# In[14]:


with parallel_backend('threading'):
    nn_grid_search.fit(X,y)


# In[20]:


with parallel_backend('threading'):    
    nn_learning_curve_results["train_sizes"], nn_learning_curve_results["train_scores"], nn_learning_curve_results["test_scores"], nn_learning_curve_results["fit_times"], _ = learning_curve(nn_grid_search.best_estimator_, X, y, cv=cross_validation, verbose=3, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True) 


# In[23]:


with parallel_backend('threading'):
    nn_validation_curve_results["train_scores"], nn_validation_curve_results["test_scores"] = validation_curve(nn_grid_search.best_estimator_, X=Validation_X, y=validation_y, n_jobs=n_jobs, verbose=3, cv=cross_validation, param_name="batch_size", param_range=[i for i in range(1, 50)])
    


# In[58]:


plt.close("all")

with plt.style.context('ggplot'):
    nn_train_scores_mean = np.mean(nn_learning_curve_results["train_scores"], axis=1)
    nn_train_scores_std = np.std(nn_learning_curve_results["train_scores"], axis=1)
    nn_test_scores_mean = np.mean(nn_learning_curve_results["test_scores"], axis=1)
    nn_test_scores_std = np.std(nn_learning_curve_results["test_scores"], axis=1)
    
    fig0, ax0 = plt.subplots()
    ax0.set_xlabel("Percent of Training Set")
    ax0.set_ylabel("Accuracy (%)", color='tab:orange')
    ax0.set_title("{} {} Learning Curve".format(nn_segment, dataset))
    ax0.tick_params(axis='y', labelcolor="black")
    ax0.set_ylim(0, 1.1)

    lw = 2
    plt.plot([i for i in range(50)], nn_train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between([i for i in range(50)], nn_train_scores_mean - nn_train_scores_std,
                     nn_train_scores_mean + nn_train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot([i for i in range(50)], nn_test_scores_mean, label="Testing score",
                 color="navy", lw=lw)
    plt.fill_between([i for i in range(50)], nn_test_scores_mean - nn_test_scores_std,
                     nn_test_scores_mean + nn_test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig("{}_{}_Learning_Curve.png".format(nn_segment, dataset))
    plt.show()
    
   


# In[59]:


with plt.style.context('ggplot'):
    nn_train_scores_mean = np.mean(nn_validation_curve_results["train_scores"], axis=1)
    nn_train_scores_std = np.std(nn_validation_curve_results["train_scores"], axis=1)
    nn_test_scores_mean = np.mean(nn_validation_curve_results["test_scores"], axis=1)
    nn_test_scores_std = np.std(nn_validation_curve_results["test_scores"], axis=1)

    plt.title("Validation Curve with {} \n {}".format(nn_segment, dataset))
    plt.xlabel("{}".format("batch_size"))
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot([i for i in range(49)], nn_train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between([i for i in range(49)], nn_train_scores_mean - nn_train_scores_std,
                     nn_train_scores_mean + nn_train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot([i for i in range(49)], nn_test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between([i for i in range(49)], nn_test_scores_mean - nn_test_scores_std,
                     nn_test_scores_mean + nn_test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig("{}_{}_Validation_Curve.png".format(nn_segment, dataset))
    plt.show()


# In[9]:


knn_segment = "K-Nearest Neighbors"
knn_estimator = KNeighborsClassifier()
knn_grid_search = GridSearchCV(knn_estimator, param_grid=knn_params, n_jobs=n_jobs, verbose=verbose_int)
knn_learning_curve_results = {"train_sizes": None, "train_scores": None, "test_scores": None, 
                              "fit_times": None}
knn_validation_curve_results = {"train_scores": None, "test_scores": None}


# In[10]:


with parallel_backend('threading'):
    knn_grid_search.fit(X,y)


# In[37]:


with parallel_backend('threading'):
        knn_learning_curve_results["train_sizes"], knn_learning_curve_results["train_scores"], knn_learning_curve_results["test_scores"], knn_learning_curve_results["fit_times"], _ = learning_curve(knn_grid_search.best_estimator_, X, y, cv=3, verbose=3, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)


# In[38]:


with parallel_backend('threading'):
        knn_validation_curve_results["train_scores"], knn_validation_curve_results["test_scores"] = validation_curve(knn_grid_search.best_estimator_, X=Validation_X, y=validation_y, n_jobs=n_jobs, cv=3, verbose=3, param_name="leaf_size", param_range=[i for i in range(2, 52)])

plt.close("all")


# In[56]:



with plt.style.context('ggplot'):
    knn_train_scores_mean = np.mean(knn_learning_curve_results["train_scores"], axis=1)
    knn_train_scores_std = np.std(knn_learning_curve_results["train_scores"], axis=1)
    knn_test_scores_mean = np.mean(knn_learning_curve_results["test_scores"], axis=1)
    knn_test_scores_std = np.std(knn_learning_curve_results["test_scores"], axis=1)
    
    fig0, ax0 = plt.subplots()
    ax0.set_xlabel("Percent of Training Set")
    ax0.set_ylabel("Accuracy (%)", color='tab:orange')
    ax0.set_title("{} {} Learning Curve".format(knn_segment, dataset))
    ax0.tick_params(axis='y', labelcolor="black")
    ax0.set_ylim(0, 1.1)

    lw = 2
    plt.plot([i for i in range(50)], knn_train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between([i for i in range(50)], knn_train_scores_mean - knn_train_scores_std,
                     knn_train_scores_mean + knn_train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot([i for i in range(50)], knn_test_scores_mean, label="Testing score",
                 color="navy", lw=lw)
    plt.fill_between([i for i in range(50)], knn_test_scores_mean - knn_test_scores_std,
                     knn_test_scores_mean + knn_test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig("{}_{}_Learning_Curve.png".format(knn_segment, dataset))
    plt.show()
    
plt.close("all")
  


# In[57]:


with plt.style.context('ggplot'):
    knn_train_scores_mean = np.mean(knn_validation_curve_results["train_scores"], axis=1)
    knn_train_scores_std = np.std(knn_validation_curve_results["train_scores"], axis=1)
    knn_test_scores_mean = np.mean(knn_validation_curve_results["test_scores"], axis=1)
    knn_test_scores_std = np.std(knn_validation_curve_results["test_scores"], axis=1)

    plt.title("Validation Curve with {} \n {}".format(knn_segment, dataset))
    plt.xlabel("{}".format("leaf_size"))
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot([i for i in range(50)], knn_train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between([i for i in range(50)], knn_train_scores_mean - knn_train_scores_std,
                     knn_train_scores_mean + knn_train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot([i for i in range(50)], knn_test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between([i for i in range(50)], knn_test_scores_mean - knn_test_scores_std,
                     knn_test_scores_mean + knn_test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig("{}_{}_Validation_Curve.png".format(knn_segment, dataset))
    plt.show() 


# In[6]:


boost_segment = "Gradient Boosted Decision Trees"
boost_estimator = GradientBoostingClassifier()
boost_grid_search = GridSearchCV(boost_estimator, param_grid=boosting_params, n_jobs=n_jobs, verbose=verbose_int)
boost_learning_curve_results = {"train_sizes": None, "train_scores": None, "test_scores": None, 
                                "fit_times": None}
boost_validation_curve_results = {"train_scores": None, "test_scores": None}


# In[7]:


with parallel_backend('threading'):
    boost_grid_search.fit(X,y)


# In[47]:


with parallel_backend('threading'):    
    boost_learning_curve_results["train_sizes"], boost_learning_curve_results["train_scores"], boost_learning_curve_results["test_scores"], boost_learning_curve_results["fit_times"], _ = learning_curve(boost_grid_search.best_estimator_, X, y, cv=2, verbose=3, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)
    


# In[48]:


with parallel_backend('threading'):    
    boost_validation_curve_results["train_scores"], boost_validation_curve_results["test_scores"] = validation_curve(boost_grid_search.best_estimator_, X=Validation_X, y=validation_y, n_jobs=n_jobs, cv=2, param_name="max_leaf_nodes", verbose=3, param_range=[i for i in range(1, 50)])

plt.close("all")


# In[51]:


with plt.style.context('ggplot'):
    boost_train_scores_mean = np.mean(boost_learning_curve_results["train_scores"], axis=1)
    boost_train_scores_std = np.std(boost_learning_curve_results["train_scores"], axis=1)
    boost_test_scores_mean = np.mean(boost_learning_curve_results["test_scores"], axis=1)
    boost_test_scores_std = np.std(boost_learning_curve_results["test_scores"], axis=1)
    
    fig0, ax0 = plt.subplots()
    ax0.set_xlabel("Percent of Training Set")
    ax0.set_ylabel("Accuracy (%)", color='tab:orange')
    ax0.set_title("{} {} Learning Curve".format(boost_segment, dataset))
    ax0.tick_params(axis='y', labelcolor="black")
    ax0.set_ylim(0, 1.1)

    lw = 2
    plt.plot([i for i in range(50)], boost_train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between([i for i in range(50)], boost_train_scores_mean - boost_train_scores_std,
                     boost_train_scores_mean + boost_train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot([i for i in range(50)], boost_test_scores_mean, label="Testing score",
                 color="navy", lw=lw)
    plt.fill_between([i for i in range(50)], boost_test_scores_mean - boost_test_scores_std,
                     boost_test_scores_mean + boost_test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig("{}_{}_Learning_Curve.png".format(boost_segment, dataset))
    plt.show()
    
plt.close("all")
   


# In[53]:


with plt.style.context('ggplot'):
    boost_train_scores_mean = np.mean(boost_validation_curve_results["train_scores"], axis=1)
    boost_train_scores_std = np.std(boost_validation_curve_results["train_scores"], axis=1)
    boost_test_scores_mean = np.mean(boost_validation_curve_results["test_scores"], axis=1)
    boost_test_scores_std = np.std(boost_validation_curve_results["test_scores"], axis=1)

    plt.title("Validation Curve with {}\n {}".format(boost_segment, dataset))
    plt.xlabel("{}".format("max_leaf_nodes"))
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot([i for i in range(49)], boost_train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between([i for i in range(49)], boost_train_scores_mean - boost_train_scores_std,
                     boost_train_scores_mean + boost_train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot([i for i in range(49)], boost_test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between([i for i in range(49)], boost_test_scores_mean - boost_test_scores_std,
                     boost_test_scores_mean + boost_test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig("{}_{}_Validation_Curve.png".format(boost_segment, dataset))
    plt.show()


# In[ ]:




