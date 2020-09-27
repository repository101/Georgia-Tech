#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time

import mlrose_hiive
import numpy as np
from mlrose_hiive.neural.activation import relu
from mlrose_hiive.runners import NNGSRunner
from sklearn.metrics import accuracy_score

import ml_util_assignment_2 as utl

save_directory = "figures/NeuralNetwork"
model_name = "NeuralNetwork"

folders = ["figures/NeuralNetwork/Complexity_Analysis",
           "figures/NeuralNetwork/Grid_Search_Results",
           "figures/NeuralNetwork/Learning_Curves",
           "figures/NeuralNetwork/Confusion_Matrix",
           "figures/NeuralNetwork/Metrics"]

directories = {
	"Save Directory": "figures/NeuralNetwork",
	"Initial Complexity Analysis": "figures/NeuralNetwork/Initial Complexity Analysis",
	"Grid Search Results": "figures/NeuralNetwork/Grid Search Results",
	"Learning Curves": "figures/NeuralNetwork/Learning Curves",
	"Final Complexity Analysis": "figures/NeuralNetwork/Final Complexity Analysis"
}

Random_Number = 42
TESTING = True
cv = 5
n_jobs = -1
np.random.seed(42)
# get_ipython().system('pip install pyarrow')


# In[2]:


gathered_data = utl.setup(["MNIST"])
# gathered_data_fashion = utl.setup(["Fashion-MNIST"])
train_X, train_y, valid_X, valid_y, test_X, test_y = utl.split_data(gathered_data["MNIST"]["X"],
                                                                    gathered_data["MNIST"]["y"], minMax=True,
                                                                    oneHot=True)
# fashion_train_X, fashion_train_y, fashion_valid_X, fashion_valid_y, fashion_test_X, fashion_test_y = utl.split_data(
#     gathered_data_fashion["Fashion-MNIST"]["X"], 
#     gathered_data_fashion["Fashion-MNIST"]["y"], minMax=True)


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

# # Supervised Learning Neural Network Parameters

# In[4]:


hidden_layer_sizes = [[100]]  # was (100,) in previous but mlrose needs a list

random_state = 42

val = 3000
pred_val = 3000

activation = 'relu'

max_iterations = 5000

# In[ ]:


# type(train_y)
#
#
# # In[ ]:
#
#
# train_X.iloc[:val, :].shape


# In[13]:

for rate in 10. * np.arange(1, 8, 1):
	for iterations in 10. ** np.arange(3, 4, 1):
		temp_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[100], activation='relu', algorithm='random_hill_climb',
		                                     max_iters=iterations, bias=True, is_classifier=True, learning_rate=rate,
		                                     early_stopping=True, clip_max=1000, max_attempts=100, restarts=3,
		                                     random_state=42, schedule=mlrose_hiive.GeomDecay(init_temp=100))
		print(f"Learning Rate: {rate}\n\tIterations: {iterations}")
		start_time = time.time()
		temp_nn.fit(train_X.iloc[:val, :], train_y.iloc[:val, :])
		end_time = time.time()
		elapsed_time = end_time - start_time
		print(f"\t\tElapsed Time: {elapsed_time}s")
		y_pred = temp_nn.predict(train_X.iloc[:val, :])
		y_train_acc = accuracy_score(np.argmax(train_y.iloc[:val, :].to_numpy(), axis=1), np.argmax(y_pred, axis=1))
		print(f"\t\tTraining Accuracy: {y_train_acc}")
		print(y_train_acc)

grid_search_parameters = ({
	'max_iters': [200],
	'learning_rate': 10. ** np.arange(-5, 2, 1),  # nn params
})

nn_rhc_gs = NNGSRunner(x_train=train_X.iloc[:val, :].to_numpy(),
                       y_train=train_y.iloc[:val, :].to_numpy(),
                       x_test=test_X.iloc[:val, :].to_numpy(),
                       y_test=test_y.iloc[:val, :].to_numpy(),
                       experiment_name='nn_rhc_test',
                       activation=[relu],
                       algorithm=mlrose_hiive.algorithms.rhc.random_hill_climb,
                       grid_search_scorer_method=accuracy_score,
                       grid_search_parameters=grid_search_parameters,
                       iteration_list=[10, 50, 100, 500, 1000, 2000],
                       is_classifier=[True],
                       hidden_layer_sizes=hidden_layer_sizes,
                       bias=True,
                       early_stopping=True,
                       clip_max=100,
                       restarts=[1],
                       max_attempts=10,
                       generate_curves=True,
                       n_jobs=6,
                       cv=5,
                       seed=random_state,
                       output_directory=f"{os.getcwd()}/results/")
rhc_res = nn_rhc_gs.run()
print()
a = rhc_res[3].best_estimator_

y_pred = a.predict(train_X.iloc[:val, :])
y_train_acc = accuracy_score(np.argmax(train_y.iloc[:val, :].to_numpy(), axis=1), np.argmax(y_pred, axis=1))
print(y_train_acc)
print()
# In[14]:


rhc_res

# # Randomized Hill Climb Optimization

# In[ ]:
#
#
# nn_model_random_hill = mlr_h.NeuralNetwork(hidden_nodes = hidden_layer_sizes, activation = activation,
#                                             algorithm = 'random_hill_climb', max_iters = max_iterations,
#                                             bias = True, is_classifier = True, learning_rate = 10.0,
#                                             early_stopping = True, clip_max = 5, max_attempts = 1000,
#                                             random_state = random_state, curve=True)
#
#
# # In[ ]:
#
#
# nn_model_random_hill.fit(fashion_train_X.iloc[:val, :], fashion_train_y.iloc[:val])
#
#
# # In[ ]:
#
#
# y_pred = nn_model_random_hill.predict(fashion_train_X.iloc[:val, :])
# y_train_acc = accuracy_score(np.argmax(fashion_train_y.iloc[:val,:].to_numpy(), axis=1), np.argmax(y_pred, axis=1))
# print(y_train_acc)
#
#
# # In[ ]:
#
#
# np.argmax(y_pred, axis=1)
#
#
# # In[ ]:
#
#
# np.argmax(fashion_train_y.iloc[:val,:].to_numpy(), axis=1)
#
#
# # In[ ]:
#
#
# print(nn_model_random_hill.node_list)
#
#
# # In[ ]:
#
#
# fig, ax = plt.subplots()
# ax.plot(np.arange(0,max_iterations),nn_model_random_hill.fitness_curve)
#
#
# # In[ ]:
#
#
# fig, ax = plt.subplots()
# ax.plot(np.arange(0,max_iterations),nn_model_random_hill.fitness_curve)
#
#
# # In[ ]:
#
#
# fig, ax = plt.subplots()
# ax.plot(np.arange(0,max_iterations),nn_model_random_hill.fitness_curve)
#
#
# # In[ ]:
#
#
# fig, ax = plt.subplots()
# ax.plot(np.arange(0,1000),nn_model_random_hill.fitness_curve)
#
#
# # # Simulated Annealing Optimization
#
# # In[ ]:
#
#
#
#
#
# # # Genetic Algortihm Optimization
#
# # In[ ]:
#
#
#
#
