#!/usr/bin/env python
# coding: utf-8

# # Initialize and Setup

# In[1]:


import os
import time

import numpy as np
from sklearn.neural_network import MLPClassifier

import ro_ml_util as utl

save_directory = "figures/NeuralNetwork"
model_name = "Neural Network"

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
n_jobs = 6
TESTING = False
cv = 5
np.random.seed(42)
# get_ipython().system('pip install pyarrow')


# In[2]:


gathered_data_fashion = utl.setup(["Fashion-MNIST"])

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


mnist_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
fashion_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                 "Ankle boot"]


# In[5]:


# Hidden Layer Sizes (100), suggests there is an input layer connected to a hidden layer with 100 nodes. (100,100) suggests
#    that there is an input layer connected to a hidden layer with 100 nodes, which is then connected to another hidden
#    layer with 100 nodes

# Solver ~ 'adam': stochastic gradient based optimizer (works well on larger datasets)
#          'lbfgs': works well on smaller datasets
#          'sgd': stochastic gradient descent

# Warmstart allows to train multiple times on same dataset using results from previous training session


if TESTING:
    val = 600
    pred_val = 600
    train_sizes = np.linspace(0.05, 1.0, 5)
else:
    val = 1000
    pred_val = 1000
    train_sizes = np.linspace(0.05, 1.0, 5)

import mlrose_hiive as mlrose_hiive
from mlrose_hiive.algorithms.decay import ArithDecay, ExpDecay, GeomDecay


# # Final Learning Curve

# In[6]:
gathered_data_fashion = utl.setup(["Fashion-MNIST"])

fashion_train_X, fashion_train_y, fashion_valid_X, fashion_valid_y, fashion_test_X, fashion_test_y = utl.split_data(
    gathered_data_fashion["Fashion-MNIST"]["X"],
    gathered_data_fashion["Fashion-MNIST"]["y"], oneHot=True, minMax=True)
val = 1000
pred_val = 1000
train_sizes = np.linspace(0.05, 1.0, 10)
a = np.argmax(fashion_train_y.iloc[:val, :], axis=1)
temp_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[100], activation='relu', pop_size=200,
                                     mutation_prob=0.1, schedule=GeomDecay(), learning_rate=10,
                                     max_iters=1000, algorithm='random_hill_climb', bias=True,
                                     is_classifier=True, early_stopping=True, clip_max=1e+5,
                                     max_attempts=10, restarts=0, random_state=45, curve=True)
utl.plot_learning_curve(estimator=temp_nn, title="Randomized Hill Climb", test_X=fashion_test_X.iloc[:val, :],
                        test_y=fashion_test_y.iloc[:val, :], train_X=fashion_train_X.iloc[:val, :],
                        train_y=np.argmax(fashion_train_y.iloc[:val, :], axis=1),
                        cv=5, f_name="IDK",
                        train_sizes=[], folder='NeuralNetwork', save_individual=True, TESTING=True,
                        backend='loky', n_jobs=6, extra_name="Final_Learning_Curve", confusion=True)



start_time = time.time()
results = []
for i in range(1, 2, 1):
    print(f"Working on learning curve: {i}")
    res = {"plt": None,
           "nn_results": None,
           "cv_results": None}

    temp_train_X = fashion_train_X.iloc[:val, :]
    temp_train_y = fashion_train_y.iloc[:val]
    title = f"{model_name} Fashion MNIST\n Learning Curve"
    f_name = f"{model_name}_Fashion_MNIST"
    optimized_nn = MLPClassifier(hidden_layer_sizes=(40, ), solver='adam', max_iter=400, alpha=1.0)
    
    res['nn_results'], res['cv_results'] = utl.plot_learning_curve(estimator=optimized_nn, title=title,
                                                                   test_X=fashion_test_X,
                                                                   test_y=fashion_test_y,
                                                                   train_X=temp_train_X, train_y=temp_train_y, cv=cv,
                                                                   f_name=f_name, train_sizes=train_sizes,
                                                                   folder='NeuralNetwork',
                                                                   save_individual=True, TESTING=True, backend='loky',
                                                                   n_jobs=n_jobs,
                                                                   extra_name="Final_Learning_Curve", confusion=True,
                                                                   confusion_name="Fashion_MNIST")
    results.append(res)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time}s")


# # Final Confusion Matrix on Test set "(O_o)"

# In[9]:


optimized_nn = MLPClassifier(hidden_layer_sizes=(40, ) , solver='adam', max_iter=400, alpha=1.0)
optimized_nn.fit(fashion_train_X.iloc[:10000, :], fashion_train_y.iloc[:10000])


# In[10]:


utl.plt_confusion_matrix(optimized_nn, fashion_test_X, fashion_test_y, directory=save_directory, fmt="d",
                          plot_width=12, plot_height=6, folder="NeuralNetwork")

