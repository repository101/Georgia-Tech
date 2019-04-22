## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to detect eye state

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

######################################### Reading and Splitting the Data ###############################################
# XXX
# TODO: Read in all the data. Replace the 'xxx' with the path to the data set.
# XXX


# Separate out the x_data and y_data.


# The random state to use while splitting the data.


# XXX
# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
# X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, shuffle=True, random_state=random_state)
# XXX


# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
# Lin_Reg_Model = LinearRegression().fit(X_train, y_train)
# XXX


# XXX
# TODO: Test its accuracy (on the training set) using the accuracy_score method.
# lin_reg_pred_train = Lin_Reg_Model.predict(X_train).round()
# Lin_Reg_Acc_Train = round(accuracy_score(y_train, lin_reg_pred_train), 2)
# # TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# lin_reg_pred_test = Lin_Reg_Model.predict(X_test).round()
# Lin_Reg_Acc_Test = round(accuracy_score(y_test, lin_reg_pred_test), 2)
# print("\nLinear Regression Training Accuracy: ", Lin_Reg_Acc_Train)
# print("Linear Regression Testing Accuracy: ", Lin_Reg_Acc_Test)

# Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use y_predict.round() or any other method.
# XXX


# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
# Rand_Forest_Model = RandomForestClassifier(n_estimators=10)
# Rand_Forest_Model.fit(X_train, y_train)
#print(np.argsort(Rand_Forest_Model.feature_importances_))
#
# feats = {}
# for feature, importance in zip(data.columns, Rand_Forest_Model.feature_importances_):
#     feats[feature] = importance
#
# importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
# list_of_importance = [[key, val] for key, val in importances.iteritems()]
# min_import = min(importances.values)
# max_import = max(importances.values)
#
# min_column = 0
# max_column = 0
# for key, val in importances.iterrows():
#     if val['Gini-importance'] == min_import:
#         min_column = key
#     if val['Gini-importance'] == max_import:
#         max_column = key
# print("The min_column is: ", min_column)
# print("The max_column is: ", max_column)

# XXX


# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# rand_forest_pred_train = Rand_Forest_Model.predict(X_train)
# Rand_Forest_Acc_Train = round(accuracy_score(y_train, rand_forest_pred_train), 2)
# # TODO: Test its accuracy on the test set using the accuracy_score method.
# rand_forest_pred_test = Rand_Forest_Model.predict(X_test)
# Rand_Forest_Acc_Test = round(accuracy_score(y_test, rand_forest_pred_test), 2)
#
# print("\nRandom Forest Training Accuracy: ", Rand_Forest_Acc_Train)
# print("Random Forest Testing Accuracy: ", Rand_Forest_Acc_Test)
# XXX


# XXX
# TODO: Determine the feature importance as evaluated by the Random Forest Classifier.

#       Sort them in the descending order and print the feature numbers. The report the most important and the least important feature.
#       Mention the features with the exact names, e.g. X11, X1, etc.
#       Hint: There is a direct function available in sklearn to achieve this. Also checkout argsort() function in Python.
# XXX


# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# Get the training and test set accuracy values after hyperparameter tuning.
# XXX


# ############################################ Support Vector Machine ###################################################
# 3.1
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
# Scaler = StandardScaler()
# Scaler.fit(X_train)
# X_Train_Scaled = Scaler.transform(X_train)
# X_Test_Scaled = Scaler.transform(X_test)
# # TODO: Create a SVC classifier and train it.
# SVC_Model = SVC()
# Tuned_SVC_Model = SVC(gamma='scale')
# SVC_Model.fit(X_Train_Scaled, y_train)

# XXX


# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# svc_pred_train = SVC_Model.predict(X_Train_Scaled)
# SVC_Acc_Train = round(accuracy_score(y_train, svc_pred_train), 2)
# # TODO: Test its accuracy on the test set using the accuracy_score method.
# svc_pred_test = SVC_Model.predict(X_Test_Scaled)
# SVC_Acc_Test = round(accuracy_score(y_test, svc_pred_test), 2)
#
# print("\nSVC Training Accuracy: ", SVC_Acc_Train)
# print("SVC Testing Accuracy: ", SVC_Acc_Test)
# XXX



# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# svc_parameters = {'kernel': ('linear', 'rbf'), 'C': [0.001, 0.1, 10]}
# Grid_SVC = GridSearchCV(Tuned_SVC_Model, svc_parameters, cv=10, verbose=1, n_jobs=-1)
# SVC_Results = Grid_SVC.fit(X_Train_Scaled, y_train)
# print("SVC_Results: \n\nBEST PARAMS: \n", SVC_Results.best_params_)
# print("\n\nBEST SCORE: \n", round(SVC_Results.best_score_, 2))
# print("\n\nSVC mean_test_score: \n", SVC_Results.cv_results_['mean_test_score'])
# print("\n\nSVC mean_train_score: \n", SVC_Results.cv_results_['mean_train_score'])
# print("\n\nSVC mean_fit_time: \n", SVC_Results.cv_results_['mean_fit_time'])
# print()
# rand_forest_parameters = {'n_estimators': [10, 100, 1000], 'max_depth': [10, 100, 1000]}
# Tuned_Rand_Forest_Model = RandomForestClassifier()
# Grid_Rand_Forest = GridSearchCV(Tuned_Rand_Forest_Model, rand_forest_parameters, cv=10, verbose=1, n_jobs=-1)
# Random_Forest_Results = Grid_Rand_Forest.fit(X_Train_Scaled, y_train)
# print("\nRandom Forest Results: \n\nBEST PARAMS: \n", Random_Forest_Results.best_params_)
# print("\n\nBEST SCORE: \n", round(Random_Forest_Results.best_score_, 2))
# print("\n\nRANDOM FOREST RESULTS: \n", Random_Forest_Results.cv_results_)
# print()
# # Get the training and test set accuracy values after hyperparameter tuning.
# # XXX


# XXX
# TODO: Calculate the mean training score, mean testing score and mean fit time for the 
# best combination of hyperparameter values that you obtained in Q3.2. The GridSearchCV 
# class holds a  ‘cv_results_’ dictionary that should help you report these metrics easily.
# XXX

# ######################################### Principal Component Analysis #################################################
# XXX
# TODO: Perform dimensionality reduction of the data using PCA.
#       Set parameters n_component to 10 and svd_solver to 'full'. Keep other parameters at their default value.
#       Print the following arrays:
#       - Percentage of variance explained by each of the selected components
#       - The singular values corresponding to each of the selected components.

# pca = PCA(n_components=10, svd_solver='full')
# pca.fit(x_data)
# print("PCA Explained Variance Ratio\n", pca.explained_variance_ratio_)
# print("PCA Singular Values\n", pca.singular_values_)
# print()
# XXX



###  3.1   ###
# TODO: Read in all the data. Replace the 'xxx' with the path to the data set.
data = pd.read_csv('eeg_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100

# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, shuffle=True, random_state=random_state)

# ############################################### Linear Regression ###################################################

# TODO: Create a LinearRegression classifier and train it.
Lin_Reg_Model = LinearRegression().fit(X_train, y_train)

# TODO: Test its accuracy (on the training set) using the accuracy_score method.
lin_reg_pred_train = Lin_Reg_Model.predict(X_train).round()
Lin_Reg_Acc_Train = round(accuracy_score(y_train, lin_reg_pred_train), 2)
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
lin_reg_pred_test = Lin_Reg_Model.predict(X_test).round()
Lin_Reg_Acc_Test = round(accuracy_score(y_test, lin_reg_pred_test), 2)
print("\nLinear Regression Training Accuracy: ", Lin_Reg_Acc_Train)
print("Linear Regression Testing Accuracy: ", Lin_Reg_Acc_Test)

# ############################################### Random Forest Classifier ##############################################

# TODO: Create a RandomForestClassifier and train it.
Rand_Forest_Model = RandomForestClassifier(n_estimators=10)
Rand_Forest_Model.fit(X_train, y_train)

# TODO: Test its accuracy on the training set using the accuracy_score method.
rand_forest_pred_train = Rand_Forest_Model.predict(X_train)
Rand_Forest_Acc_Train = round(accuracy_score(y_train, rand_forest_pred_train), 2)
# TODO: Test its accuracy on the test set using the accuracy_score method.
rand_forest_pred_test = Rand_Forest_Model.predict(X_test)
Rand_Forest_Acc_Test = round(accuracy_score(y_test, rand_forest_pred_test), 2)

print("\nRandom Forest Training Accuracy: ", Rand_Forest_Acc_Train)
print("Random Forest Testing Accuracy: ", Rand_Forest_Acc_Test)

# ############################################ Support Vector Machine ###################################################
# 3.1
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
Scaler = StandardScaler()
Scaler.fit(X_train)
X_Train_Scaled = Scaler.transform(X_train)
X_Test_Scaled = Scaler.transform(X_test)
# TODO: Create a SVC classifier and train it.
SVC_Model = SVC()
SVC_Model.fit(X_Train_Scaled, y_train)
# TODO: Test its accuracy on the training set using the accuracy_score method.
svc_pred_train = SVC_Model.predict(X_Train_Scaled)
SVC_Acc_Train = round(accuracy_score(y_train, svc_pred_train), 2)
# TODO: Test its accuracy on the test set using the accuracy_score method.
svc_pred_test = SVC_Model.predict(X_Test_Scaled)
SVC_Acc_Test = round(accuracy_score(y_test, svc_pred_test), 2)
print("\nSVC Training Accuracy: ", SVC_Acc_Train)
print("SVC Testing Accuracy: ", SVC_Acc_Test)



# 3.2.1-3
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
rand_forest_parameters = {'n_estimators': [10, 100, 1000], 'max_depth': [10, 100, 1000]}
Tuned_Rand_Forest_Model = RandomForestClassifier()
Grid_Rand_Forest = GridSearchCV(Tuned_Rand_Forest_Model, rand_forest_parameters, cv=10, verbose=1, n_jobs=-1)
Random_Forest_Results = Grid_Rand_Forest.fit(X_Train_Scaled, y_train)
print("\nRandom Forest Results: \n\nBEST PARAMS: \n", Random_Forest_Results.best_params_)
print("\n\nBEST SCORE: \n", round(Random_Forest_Results.best_score_, 2))
print("\n\nRANDOM FOREST RESULTS: \n", Random_Forest_Results.cv_results_)
print()
Tuned_Rand_Forest_Model = RandomForestClassifier(n_estimators=1000, max_depth=100)
Tuned_Rand_Forest_Model.fit(X_train, y_train)

# TODO: Test its accuracy on the training set using the accuracy_score method.
tuned_rand_forest_pred_train = Tuned_Rand_Forest_Model.predict(X_train)
Tuned_Rand_Forest_Acc_Train = round(accuracy_score(y_train, tuned_rand_forest_pred_train), 2)
# TODO: Test its accuracy on the test set using the accuracy_score method.
tuned_rand_forest_pred_test = Tuned_Rand_Forest_Model.predict(X_test)
Tuned_Rand_Forest_Acc_Test = round(accuracy_score(y_test, tuned_rand_forest_pred_test), 2)

print("\nTuned Random Forest Training Accuracy: ", Tuned_Rand_Forest_Acc_Train)
print("Tuned Random Forest Testing Accuracy: ", Tuned_Rand_Forest_Acc_Test)



# 3.2.4-6
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
svc_parameters = {'kernel': ('linear', 'rbf'), 'C': [0.001, 0.1, 10]}
Grid_SVC = GridSearchCV(SVC_Model, svc_parameters, cv=10, verbose=1, n_jobs=-1)
SVC_Results = Grid_SVC.fit(X_Train_Scaled, y_train)
print("SVC_Results: \n\nBEST PARAMS: \n", SVC_Results.best_params_)
print("\n\nBEST SCORE: \n", round(SVC_Results.best_score_, 2))
print("\n\nSVC mean_test_score: \n", SVC_Results.cv_results_['mean_test_score'])
print("\n\nSVC mean_train_score: \n", SVC_Results.cv_results_['mean_train_score'])
print("\n\nSVC mean_fit_time: \n", SVC_Results.cv_results_['mean_fit_time'])
Tuned_SVC_Model = SVC(gamma='scale', kernel='rbf', C=10)
print()
Tuned_SVC_Model.fit(X_Train_Scaled, y_train)
# TODO: Test its accuracy on the training set using the accuracy_score method.
tuned_svc_pred_train = Tuned_SVC_Model.predict(X_Train_Scaled)
Tuned_SVC_Acc_Train = round(accuracy_score(y_train, tuned_svc_pred_train), 2)
# TODO: Test its accuracy on the test set using the accuracy_score method.
tuned_svc_pred_test = Tuned_SVC_Model.predict(X_Test_Scaled)
Tuned_SVC_Acc_Test = round(accuracy_score(y_test, tuned_svc_pred_test), 2)
print("\nTuned SVC Training Accuracy: ", Tuned_SVC_Acc_Train)
print("Tuned SVC Testing Accuracy: ", Tuned_SVC_Acc_Test)



# TODO: Determine the feature importance as evaluated by the Random Forest Classifier.
# 3.4 Feature Importance - WITH THE MODEL TRAINED IN Q 3.1
feats = {}
for feature, importance in zip(data.columns, Rand_Forest_Model.feature_importances_):
    feats[feature] = importance

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
list_of_importance = [[key, val] for key, val in importances.iteritems()]
min_import = min(importances.values)
max_import = max(importances.values)

min_column = 0
max_column = 0
for key, val in importances.iterrows():
    if val['Gini-importance'] == min_import:
        min_column = key
    if val['Gini-importance'] == max_import:
        max_column = key
print("The min_column is: ", min_column)
print("The max_column is: ", max_column)



# ######################################### Principal Component Analysis #################################################
# 3.6
# TODO: Perform dimensionality reduction of the data using PCA.
#       Set parameters n_component to 10 and svd_solver to 'full'. Keep other parameters at their default value.
#       Print the following arrays:
#       - Percentage of variance explained by each of the selected components
#       - The singular values corresponding to each of the selected components.

pca = PCA(n_components=10, svd_solver='full')
pca.fit(x_data)
explained_variance_list = list(pca.explained_variance_ratio_)
rounded_explained_variance = [round(x, 6) for x in explained_variance_list]
print("PCA Explained Variance Ratio\n", explained_variance_list)

singular_val_list = list(pca.singular_values_)
rounded_singular_values = [round(x, 2) for x in singular_val_list]
print("PCA Singular Values\n", rounded_singular_values)
print()