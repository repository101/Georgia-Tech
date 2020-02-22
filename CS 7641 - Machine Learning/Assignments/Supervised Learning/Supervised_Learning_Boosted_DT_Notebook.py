#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import parallel_backend
from timeit import default_timer as timer
from LoadData import LoadData


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


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
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                                          train_sizes=train_sizes, return_times=True)
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

cwd = pathlib.Path().absolute()
if DataSetName == "MNIST":
    training_data_path = "{}/small-mnist-train-data.csv".format(cwd)
    testing_data_path = "{}/small-mnist-test-data.csv".format(cwd)
else:
    training_data_path = "{}/small-fashion-mnist-train-data.csv".format(cwd)
    testing_data_path = "{}/small-fashion-mnist-test-data.csv".format(cwd)

training_labels, training_data, _ = LoadData(training_data_path, normalize=NORMALIZE_DATA)
testing_labels, testing_data, _ = LoadData(testing_data_path, normalize=NORMALIZE_DATA)

Scaler = StandardScaler().fit(training_data)

training_data = Scaler.transform(training_data)
testing_data = Scaler.transform(testing_data)

fig, axes = plt.subplots(3, 2, figsize=(10, 15))

title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(estimator, title, training_data, training_labels, axes=axes[:, 0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, training_data, training_labels, axes=axes[:, 1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()


# In[ ]:


cwd = pathlib.Path().absolute()
if DataSetName == "MNIST":
    training_data_path = "{}/mnist-train-data.csv".format(cwd)
    testing_data_path = "{}/mnist-test-data.csv".format(cwd)
else:
    training_data_path = "{}/fashion-mnist-train-data.csv".format(cwd)
    testing_data_path = "{}/fashion-mnist-test-data.csv".format(cwd)


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


classifier_list = []
runtime = [0.0]
accuracy = [0.0]


# In[ ]:


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier


# In[ ]:


clf = HistGradientBoostingClassifier(max_depth=3)
clf.fit(training_data[:int((60000 * 0.1)), :], training_labels[:int((60000 * 0.1))])


# In[ ]:


clf.validation_score_


# In[ ]:


with parallel_backend('threading'):
    clf = HistGradientBoostingClassifier(max_depth=3)
#     clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, verbose=3, learning_rate=0.1)
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


print(accuracy)


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





# In[ ]:


colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]

run = runtime
acc = accuracy


with plt.style.context('ggplot'):
    fig0, ax0 = plt.subplots()
    ax0.set_xlabel("Percent of Training Set")
    ax0.set_ylabel("Accuracy (%)", color='tab:orange')
    ax0.set_title("Accuracy vs Training Set Size vs Training Time {} \n {}".format("Boosted-Hist", DataSetName))
    ax0.tick_params(axis='y', labelcolor="black")
    ax0.set_ylim(0, 1.1)
    ax3 = ax0.twinx()
    ax3.set_ylabel("Training Time (s)", color="tab:blue")
    ax3.set_ylim(0, max(max(runtime), max(runtime)) + 10)
    ax3.tick_params(axis='y', labelcolor="black")
    for i in range(1):        
        ax0.plot([i for i in range(11)], acc, colors[i], marker='o', label="Boosted-Hist Accuracy")
        ax3.plot([i for i in range(11)], run, colors[i+1], marker="1", label="{} training-time".format("Boosted-Hist Training Time"))
    fig0.tight_layout()
    directory = "{}/Training_{}_{}_Hist_Set_Size_Impact_vs_Training_Time.png".format(cwd, "Boosted", DataSetName)
    plt.savefig(directory)

