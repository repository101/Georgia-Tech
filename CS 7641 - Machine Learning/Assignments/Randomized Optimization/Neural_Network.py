#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import time

import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import pathlib

from LoadData import LoadData
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict, GridSearchCV, train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.neural_network import MLPClassifier
from sklearn.utils import parallel_backend
from sklearn.preprocessing import StandardScaler


random_state = 7
start = 10
stop = 101
step = 10
size = 20
problem_name = "NeuralNetwork"
plt.style.use('ggplot')


# In[7]:


def plot_learning_curve(estimator, title, X, y, exp_score=0.8, 
                        ylim=None, cv=None, n_jobs=None, 
                        train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    plt.xlim(0,y.size)
    with parallel_backend('threading'):
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                                              train_sizes=train_sizes,
                                                                              return_times=True, random_state=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.axhline(y=exp_score, color='b', linestyle='-', label="Acceptable score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")

    plt.show()


def plot_validation_curve(estimator,X, y, param_name, param_range, 
                          cv, n_jobs, title, xlabel, ylim = None, scoring='f1'):     
    with parallel_backend('threading'):
        train_scores, test_scores = validation_curve(
            estimator, X, y, param_name, param_range, cv=cv,
            scoring=scoring, n_jobs=n_jobs)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("F1 Score")

    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)    
    plt.plot(param_range,train_scores_mean-test_scores_mean, label = 'Error'
                    , color="red", lw=lw)

    plt.legend(loc="best")

    plt.show()
    

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.colors.BASE_COLORS):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    plt.show()
    

def plot_roc_curve(clf, X_test, y_test, title, dfunc=False):
    
    if dfunc:
        y_score = clf.decision_function(X_test)
    else:
        y_score = clf.predict_proba(X_test)[:, 1]
       
    fpr, tpr, threshold = roc_curve(y_test, y_score)      
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = {0:0.2f})'
                             ''.format(roc_auc), linewidth=2.5)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    plt.show()

    optimal_idx = np.argmin(np.sqrt(np.square(1-tpr) + np.square(fpr)))
    optimal_threshold = threshold[optimal_idx]
    print("Optimal Threshold: ")
    print(optimal_threshold)    

    
def plot_pr_curve(clf, X_test, y_test, title, dfunc=False):
    if dfunc:
        y_score = clf.decision_function(X_test)
    else:
        y_score = clf.predict_proba(X_test)[:, 1]
    
    with parallel_backend('threading'):
        average_precision = average_precision_score(y_test, y_score)
        precision, recall, threshold = precision_recall_curve(y_test, y_score)

    plt.plot(recall, precision, label='Precision Recall curve (AP = {0:0.2f})'
                             ''.format(average_precision),
             linewidth=2.5)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
       
    max_f1 = 0
    for r, p, t in zip(recall, precision, threshold):
        if p + r == 0: continue
        if (2 * p * r) / (p + r) > max_f1:
            max_f1 = (2 * p * r) / (p + r)
            max_f1_threshold = t

    print("Optimal Threshold: ") 
    print(max_f1_threshold)
    print("Maximized F1 score: ") 
    print(max_f1)


def plot_precision_recall_vs_threshold(clf, X_test, y_test, title, dfunc=False):
    if dfunc:
        y_score = clf.decision_function(X_test)
    else:
        y_score = clf.predict_proba(X_test)[:, 1]

    with parallel_backend('threading'):
        average_precision = average_precision_score(y_test, y_score)
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_score)
    
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.legend(loc='best')
    


# ### Load Dataset

# In[3]:


cwd = pathlib.Path().absolute()
training_data_path = "{}/mnist-train-data.csv".format(cwd)
testing_data_path = "{}/mnist-test-data.csv".format(cwd)


training_labels, training_data, training_combined = LoadData(training_data_path, normalize=True)
testing_labels, testing_data, testing_combined = LoadData(testing_data_path, normalize=True)


# ### Neural Network - Randomized Optimization

# In[4]:



# Random Hill Climbing
random_hill_climbing_NN = mlrose.NeuralNetwork(hidden_nodes=[784,100], activation='relu',
                                               algorithm='random_hill_climb', max_iters=100,
                                               restarts=90, bias=True, is_classifier=True, 
                                               learning_rate=1, max_attempts= 50, mutation_prob=0.1, 
                                               pop_size=100, random_state=random_state, curve=True)

# Genetic Algorithm
genetic_algorithm_NN = mlrose.NeuralNetwork(hidden_nodes=[784,100], activation='relu',
                                            algorithm='genetic_alg', max_iters=100, pop_size=1000,
                                            bias=True, is_classifier=True, learning_rate=1, mutation_prob=0.9,
                                            max_attempts=50, random_state=random_state, curve=True)


# Simulated Annealing

simulated_annealing_NN = mlrose.NeuralNetwork(hidden_nodes=[784,100], activation='relu',
                                              algorithm='simulated_annealing', max_iters=100,
                                              schedule=mlrose.ExpDecay(),bias=True, 
                                              is_classifier=True, learning_rate=1, max_attempts=50, 
                                              mutation_prob=0.1, pop_size=100, random_state=random_state, curve=True)
simulated_annealing_NN.curve

# MIMIC


# In[8]:

with parallel_backend('threading'):
    random_hill_climbing_NN.fit(testing_data, testing_labels)
    plot_learning_curve(simulated_annealing_NN, "Random Hill Climbing - Learning Curve", training_data, training_labels, cv=5)

