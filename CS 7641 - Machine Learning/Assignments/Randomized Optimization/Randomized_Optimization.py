#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle

import mlrose_hiive
import numpy as np
import time

import ro_ml_util as utl

SEED = 42


# problem, folder = utl.determine_problem(prob_name="knapsack", size="m", maximize=True, SEED=42)
# #
# with open(f"{os.getcwd()}/{folder}/All_SA_Results.pkl", "rb") as input_file:
#     rhc_parameters = pickle.load(input_file)
#     input_file.close()
# print()
#
# with open(f"{os.getcwd()}/{folder}/SimulatedAnnealing_FINAL_Parameters_l_l_l.pkl", "rb") as input_file:
#     sa_parameters = pickle.load(input_file)
#     input_file.close()
#
# with open(f"{os.getcwd()}/{folder}/GeneticAlgorithm_FINAL_Parameters_l_l_l.pkl", "rb") as input_file:
#     ga_parameters = pickle.load(input_file)
#     input_file.close()
#
# with open(f"{os.getcwd()}/{folder}/MIMIC_FINAL_Parameters_l_l_l.pkl", "rb") as input_file:
#     mimic_parameters = pickle.load(input_file)
#     input_file.close()
#
# _results = {"RHC": rhc_parameters, "SA": sa_parameters, "GA": ga_parameters, "MIMIC": mimic_parameters}
# print()
iteration_list = np.concatenate((10 ** np.arange(1, 4, 1), (10 ** np.arange(1, 4, 1)) * 5)).tolist()

rhc_params = {"restart_list": (2 ** np.arange(6, 9, 1)).tolist(),
              "iterations": (2 ** np.arange(6, 9, 1)).tolist()}

sa_params = {"temperature_list": [1, 2, 4, 8, 30, 60, 100, 500, 1000],
             "iterations": np.arange(1000, 10001, 1000).tolist(),
            "max_attempts": 500}

ga_params = {"population_sizes": np.round(np.arange(200, 401, 25)).astype(np.int).tolist(),
             "iterations": np.round(2 ** np.arange(1, 9, 1)).astype(np.int).tolist(),
             "mutation_rates": np.round(np.arange(0.95, 1.0, 0.01), 2).tolist(),
            "max_attempts": 2}

mimic_params = {"population_sizes": np.round(np.arange(25, 126, 25)).astype(np.int).tolist(),
                "keep_percent_list": np.round(np.arange(0.1, 0.51, 0.1), 2).tolist(),
                "iterations": np.round(2 ** np.arange(1, 9, 1)).astype(np.int).tolist(),
               "max_attempts": 128}

# acc, time = utl.find_best_neural_network_rhc(train_limit=1000, num_iter=10, verbose=True, alt_method=True,
#                                              rhc_parameters=rhc_params)
# problem, folder = utl.determine_problem(prob_name="TSP", size="m", maximize=True, SEED=int(np.around(time.time())))
#
problem, folder = utl.determine_problem(prob_name="continuouspeaks", size="l", maximize=True, SEED=42)
utl.run_optimization_tests(prob_name="continuouspeaks", parameters={"RHC": rhc_params, "SA": sa_params,
                                                             "GA": ga_params, "MIMIC": mimic_params},
                           size="l", iterations=iteration_list, maximize=True, gridsearch=False,
                           gen_curves=True, cv=2, max_attempts=(2 ** np.arange(1, 8, 1)).tolist(), only_size=True,
                           eval_sizes=["l"])


#
# problem, folder = utl.determine_problem(prob_name="FlipFlop", size="m", maximize=True, SEED=42)
#
# with open(f"{os.getcwd()}/{folder}/All_MIMIC_Results.pkl", "rb") as input_file:
#     data = pickle.load(input_file)
#     input_file.close()
#
# utl.plot_discrete(all_results=data, folder=folder, prob_name="FlipFlop", alg_name="MIMIC")
# print()
