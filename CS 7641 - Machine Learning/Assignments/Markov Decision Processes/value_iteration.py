import copy
import os
import sys
import time

import numpy as np

from util import get_time_wrapper, extract_policy, evaluate_policy


def eval_state_action(V, s, a, env, gamma=0.99):
	"""
	:param V:
	:param s:
	:param a:
	:param env:
	:param gamma:
	:return:
	https://github.com/PacktPublishing/Reinforcement-Learning-Algorithms-with-Python/blob/master/Chapter03/frozenlake8x8_valueiteration.py
	"""
	try:
		return np.sum([p * (rew + gamma * V[next_s]) for p, next_s, rew, _ in env.P[s][a]])
	except Exception as eval_state_action_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'eval_state_action'", eval_state_action_exception)


def value_iteration(env, eps=1e-20, max_iterations=100, gamma=0.99):
	"""
	Value iteration algorithm

	https://github.com/PacktPublishing/Reinforcement-Learning-Algorithms-with-Python/blob/master/Chapter03/frozenlake8x8_valueiteration.py

	:param gamma:
	:param max_iterations:
	:param env:
	:param eps:
	:return:
	"""
	try:
		run_times = np.zeros(shape=(max_iterations,))
		total_values = np.zeros(shape=(max_iterations,))
		difference_per_iteration = np.zeros(shape=(max_iterations,))
		value_function = np.zeros(env.nS)
		iteration = 0  # just initializing to silence error
		for iteration in range(max_iterations):
			start_time = time.perf_counter()
			delta = 0
			# update the value of each state using as "policy" the max operator
			prev_v = np.copy(value_function)
			for s in range(env.nS):
				prev_v_state = value_function[s]
				value_function[s] = np.max([eval_state_action(value_function, s, a, env, gamma) for a in range(env.nA)])
				delta = max(delta, np.abs(prev_v_state - value_function[s]))
			difference_per_iteration[iteration] = np.sum(np.abs(prev_v - value_function))
			total_values[iteration] = np.sum(value_function)
			if delta <= eps:
				break
			else:
				print('Iter:', iteration, ' delta:', np.round(delta, 5))
			end_time = time.perf_counter()
			run_times[iteration] = end_time - start_time
		return {"Value_Function": value_function, "Number_Of_Iterations": iteration, "Run_Times": run_times,
				"Average_Run_Time": np.mean(run_times[:iteration]), "Total_Value": total_values,
				"Difference_Per_Iteration": difference_per_iteration}
	except Exception as value_iteration_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'value_iteration'", value_iteration_exception)


def run_value_iteration(*kwargs):
	try:
		temp_kwargs = copy.deepcopy(kwargs)
		temp_kwargs[0]["env"].reset()
		print(f"\nStarting: Value Iteration\n")
		temp_results = get_time_wrapper(value_iteration, *temp_kwargs)

		temp_optimal_policy = extract_policy(env=temp_kwargs[0]["env"],
											 value_function=temp_results["Results"]["Value_Function"],
											 gamma=temp_kwargs[0]["gamma"])
		avg_scores, scores = evaluate_policy(env=temp_kwargs[0]["env"], policy=temp_optimal_policy,
											 gamma=temp_kwargs[0]["gamma"])

		temp_results["Results"]["Optimal_Policy"] = temp_optimal_policy
		temp_results["Results"]["Policy_Scores"] = scores
		temp_results["Results"]["Average_Scores"] = avg_scores

		print(f"Value Iteration"
			  f"\n\tNumber of States: {temp_kwargs[0]['env'].nS}"
			  f"\n\tMax Iterations: {temp_kwargs[0]['max_iterations']}"
			  f"\n\tGamma: {temp_kwargs[0]['gamma']}"
			  f"\n\tRun Time: {temp_results['Elapsed Time']:.5f}s"
			  f"\n\tAverage Score: {avg_scores:.3f}"
			  f"\n\tHighest Score: {np.max(temp_results['Results']['Total_Value']):.3f}")

		return temp_results
	except Exception as run_value_iteration_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'run_value_iteration'", run_value_iteration_exception)
