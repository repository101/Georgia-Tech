import copy
import os
import sys
import time

import mlrose_hiive
import numpy as np
import pandas as pd

SEED = 42


def my_callback(**kwargs):
	print()
	return True


def get_rhc_iteration_times(iterations, problem, params, verbose=False):
	try:
		rhc_time_df = pd.DataFrame(index=iterations, columns=["RunTime"],
		                           data=np.zeros(shape=(iterations.shape[0],)))
		rhc_fitness_df = pd.DataFrame(index=iterations, columns=["Fitness"],
		                              data=np.zeros(shape=(iterations.shape[0],)))
		a = pd.DataFrame(index=iterations, data=np.zeros(shape=(iterations.shape[0], iterations.shape[0])))
		
		if not isinstance(iterations, list):
			iterations = iterations.tolist()
		
		temp_problem = copy.deepcopy(problem)

		temp_state, \
		temp_fitness, \
		temp_fitness_curve = mlrose_hiive.random_hill_climb(problem=temp_problem, max_attempts=max(iterations),
		                                                    restarts=params["Restart"], curve=True,
		                                                    random_state=SEED, max_iters=max(iterations), state_fitness_callback=my_callback,
		                                                    callback_user_info=[])
		rhc_fitness_df["Fitness"] = temp_fitness_curve
		count = 0
		print("Randomized Hill Climbing Starting")
		total_start_time = time.time()
		for iteration in iterations:
			if verbose:
				print(f"Current Iteration: {count} \t\tRemaining Iterations: {len(iterations) - count}")
			temp_problem = copy.deepcopy(problem)
			temp_start_time = time.time()
			temp_state, \
			temp_fitness, \
			temp_fitness_curve = mlrose_hiive.random_hill_climb(problem=temp_problem, max_attempts=params["Attempts"],
			                                                    restarts=params["Restart"], curve=True,
			                                                    random_state=SEED, max_iters=iteration)
			temp_end_time = time.time()
			for i in range(len(temp_fitness_curve)):
				a[i].loc[iteration] = temp_fitness_curve[i]
			temp_elapsed_time = temp_end_time - temp_start_time
			rhc_time_df["RunTime"].loc[iteration] = temp_elapsed_time
			count += 1
		total_end_time = time.time()
		total_elapsed_time = total_end_time - total_start_time
		print("Randomized Hill Climbing Ending")
		results = {
			"Fitness": rhc_fitness_df,
			"RunTimes": rhc_time_df,
			"TotalTime": total_elapsed_time
		}
		return results
	except Exception as get_rhc_iteration_times_except:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'get_rhc_iteration_times'.\n", get_rhc_iteration_times_except)
		print()


def get_ga_iteration_times(iterations, problem, params, verbose=False, max_iter=10):
	try:
		ga_time_df = pd.DataFrame(index=iterations, columns=["RunTime"],
		                          data=np.zeros(shape=(iterations.shape[0],)))
		ga_fitness_df = pd.DataFrame(index=iterations,
		                             columns=["Fitness"],
		                             data=np.zeros(shape=(iterations.shape[0],)))
		if not isinstance(iterations, list):
			iterations = iterations.tolist()
		count = 0
		temp_problem = copy.deepcopy(problem)
		
		temp_state, \
		temp_fitness, \
		temp_fitness_curve = mlrose_hiive.genetic_alg(problem=temp_problem, pop_size=params["Population Size"],
		                                              pop_breed_percent=params["Breed Percent"],
		                                              mutation_prob=params["Mutation Probability"],
		                                              random_state=SEED, max_attempts=max(iterations),
		                                              curve=True, max_iters=max(iterations))
		ga_fitness_df["Fitness"] = temp_fitness_curve
		
		print("Starting a Genetic Algorithm")
		
		total_start_time = time.time()
		for iteration in iterations:
			if verbose:
				print(f"Current Iteration: {count} \t\tRemaining Iterations: {len(iterations) - count}")
			temp_problem = copy.deepcopy(problem)
			temp_start_time = time.time()
			temp_state, \
			temp_fitness, \
			temp_fitness_curve = mlrose_hiive.genetic_alg(problem=temp_problem, pop_size=params["Population Size"],
			                                              pop_breed_percent=params["Breed Percent"],
			                                              mutation_prob=params["Mutation Probability"],
			                                              random_state=SEED, max_attempts=iteration,
			                                              curve=True)
			temp_end_time = time.time()
			temp_elapsed_time = temp_end_time - temp_start_time
			ga_time_df["RunTime"].loc[iteration] = temp_elapsed_time
			count += 1
		print("A Genetic Algorithm Ending")
		total_end_time = time.time()
		total_elapsed_time = total_end_time - total_start_time
		results = {
			"Fitness": ga_fitness_df,
			"RunTimes": ga_time_df,
			"TotalTime": total_elapsed_time
		}
		return results
	except Exception as get_ga_iteration_times_except:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'get_ga_iteration_times'.\n", get_ga_iteration_times_except)
		print()


def get_sa_iteration_times(iterations, problem, params, verbose=False):
	try:
		sa_time_df = pd.DataFrame(index=iterations, columns=["RunTime"],
		                          data=np.zeros(shape=(iterations.shape[0],)))
		sa_fitness_df = pd.DataFrame()
		if not isinstance(iterations, list):
			iterations = iterations.tolist()
		
		temp_problem = copy.deepcopy(problem)

		temp_state, \
		temp_fitness, \
		temp_fitness_curve = mlrose_hiive.simulated_annealing(problem=temp_problem, schedule=params["Decay Schedule"],
		                                                      curve=True, max_iters=max(iterations),
		                                                      max_attempts=max(iterations), random_state=SEED)
		sa_fitness_df["Fitness"] = temp_fitness_curve
		
		count = 0
		print("Simulated Annealing Starting")
		total_start_time = time.time()
		for iteration in iterations:
			if verbose:
				print(f"Current Iteration: {count} \t\tRemaining Iterations: {len(iterations) - count}")
			temp_problem = copy.deepcopy(problem)
			temp_start_time = time.time()
			temp_state, \
			temp_fitness, \
			temp_fitness_curve = mlrose_hiive.simulated_annealing(problem=temp_problem,
			                                                      schedule=params["Decay Schedule"],
			                                                      curve=True, max_iters=iteration, random_state=SEED)
			temp_end_time = time.time()
			temp_elapsed_time = temp_end_time - temp_start_time
			sa_time_df["RunTime"].loc[iteration] = temp_elapsed_time
			count += 1
		print("Simulated Annealing Ending")
		total_end_time = time.time()
		total_elapsed_time = total_end_time - total_start_time
		results = {
			"Fitness": sa_fitness_df,
			"RunTimes": sa_time_df,
			"TotalTime": total_elapsed_time
		}
		return results
	except Exception as get_sa_iteration_times_except:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'get_sa_iteration_times'.\n", get_sa_iteration_times_except)
		print()


def get_mimic_iteration_times(iterations, problem, params, verbose=False):
	try:
		mimic_time_df = pd.DataFrame(index=iterations, columns=["RunTime"],
		                             data=np.zeros(shape=(iterations.shape[0],)))
		mimic_fitness_df = pd.DataFrame(index=iterations, columns=["Fitness"],
		                                data=np.zeros(shape=(iterations.shape[0],)))
		a = pd.DataFrame(index=iterations, columns=iterations, data=np.zeros(shape=(iterations.shape[0],
		                                                                            iterations.shape[0])))
		if not isinstance(iterations, list):
			iterations = iterations.tolist()
		print("MIMIC Starting")
		count = 0
		total_start_time = time.time()
		temp_problem = copy.deepcopy(problem)
		temp_state, \
		temp_fitness, \
		temp_fitness_curve = mlrose_hiive.mimic(problem=temp_problem, pop_size=params["Population Size"],
		                                        keep_pct=params["Keep Percentage"],
		                                        curve=True, random_state=SEED, max_iters=max(iterations),
		                                        max_attempts=max(iterations))
		mimic_fitness_df["Fitness"] = temp_fitness_curve

		for iteration in iterations:
			if verbose:
				print(f"Current Iteration: {count} \t\tRemaining Iterations: {len(iterations) - count}")
			temp_problem = copy.deepcopy(problem)
			temp_start_time = time.time()
			temp_state, \
			temp_fitness, \
			temp_fitness_curve = mlrose_hiive.mimic(problem=temp_problem, pop_size=params["Population Size"],
			                                        keep_pct=params["Keep Percentage"],
			                                        curve=True, random_state=SEED, max_iters=iteration,
			                                        max_attempts=iteration)
			temp_end_time = time.time()
			temp_elapsed_time = temp_end_time - temp_start_time
			mimic_time_df["RunTime"].loc[iteration] = temp_elapsed_time
			count += 1
		
		print("MIMIC Ending")
		total_end_time = time.time()
		total_elapsed_time = total_end_time - total_start_time
		results = {
			"Fitness": mimic_fitness_df,
			"RunTimes": mimic_time_df,
			"TotalTime": total_elapsed_time
		}
		return results
	except Exception as get_mimic_iteration_times_except:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'get_mimic_iteration_times'.\n", get_mimic_iteration_times_except)
		print()
