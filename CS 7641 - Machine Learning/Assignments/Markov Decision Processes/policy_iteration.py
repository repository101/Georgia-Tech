import copy
import os
import sys
import time

import numpy as np

from util import evaluate_policy, get_time_wrapper

save_dir = "Graphs"


def compute_policy_v(env, policy, V, gamma=1.0, eps=1e-10, max_attempts=100):
	""" Iteratively evaluate the value-function under policy.
	Alternatively, we could formulate a set of linear equations in terms of v[s]
	and solve them to find the value function.

	https://learning.oreilly.com/library/view/deep-learning-with/9781838553005/27f144fb-4839-4d9b-ba9e-023191b9fd12.xhtml
	"""
	try:
		num_iter = 0
		for i in range(max_attempts):
			delta = 0
			for s in range(env.nS):
				old_v = V[s]
				V[s] = eval_state_action(V, s, policy[s], env=env, gamma=gamma)
				delta = max(delta, np.abs(old_v - V[s]))
			if delta < eps:
				break
			num_iter += 1
		return num_iter
	except Exception as compute_policy_v_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'compute_policy_v'", compute_policy_v_exception)


def policy_iteration(env, max_iterations=1000, gamma=0.99, eps=0.01):
	"""
	https://github.com/chaitanyamittal/Markov-Decision-Processes/blob/master/MDPs.ipynb
	:param gamma:
	:param env:
	:param max_iterations:
	:return:
	"""
	try:
		policy = np.random.choice(env.nA, size=env.nS)  # initialize a random policy
		value_function_iterations = np.zeros(shape=(max_iterations,))
		total_values = np.zeros(shape=(max_iterations,))
		run_times = np.zeros(shape=(max_iterations,))
		reached_convergence = float("-inf")
		V = np.zeros(env.nS)
		i = 0  # to silence error
		for i in range(max_iterations):
			start_time = time.perf_counter()
			num_iter = compute_policy_v(env, policy, V=V, gamma=gamma)
			total_values[i] = np.sum(V)
			value_function_iterations[i] = num_iter
			policy_evaluation(V, policy, env=env, eps=eps)
			policy_stable = policy_improvement(V, policy, env=env)
			if policy_stable:
				print('Policy-Iteration converged at step %d.' % (i + 1))
				reached_convergence = i + 1
				break
			end_time = time.perf_counter()
			run_times[i] = end_time - start_time
		return {"Policy": policy, "Number_Of_Iterations": reached_convergence, "Run_Times": run_times,
				"Total_Value": total_values, "Average_Run_Time": np.mean(run_times[:i])}
	except Exception as policy_iteration_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'policy_iteration'", policy_iteration_exception)


def eval_state_action(V, s, a, env, gamma=0.99):
	"""
	https://github.com/PacktPublishing/Reinforcement-Learning-Algorithms-with-Python/blob/master/Chapter03/frozenlake8x8_policyiteration.py
	:param V:
	:param s:
	:param a:
	:param env:
	:param gamma:
	:return:
	"""
	try:
		return np.sum([current_state * (reward + gamma * V[next_state]) for
					   current_state, next_state, reward, _ in env.P[s][a]])
	except Exception as eval_state_action_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'eval_state_action'", eval_state_action_exception)


def policy_evaluation(V, policy, env, eps=0.0001, max_attempts=100):
	"""
	https://github.com/PacktPublishing/Reinforcement-Learning-Algorithms-with-Python/blob/master/Chapter03/frozenlake8x8_policyiteration.py
	:param V:
	:param policy:
	:param env:
	:param eps:
	:param max_attempts:
	:return:
	"""
	try:
		delta = 0
		for _ in range(max_attempts):
			delta = 0
			# loop over all states
			for s in range(env.nS):
				old_v = V[s]
				# update V[s] using the Bellman equation
				V[s] = eval_state_action(V, s, policy[s], env=env)
				delta = max(delta, np.abs(old_v - V[s]))
			if delta < eps:
				break

		return V, delta
	except Exception as policy_evaluation_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'policy_evaluation'", policy_evaluation_exception)


def policy_improvement(V, policy, env):
	"""
	https://github.com/PacktPublishing/Reinforcement-Learning-Algorithms-with-Python/blob/master/Chapter03/frozenlake8x8_policyiteration.py
	:param gamma:
	:param V:
	:param policy:
	:param env:
	:return:
	"""
	try:
		policy_stable = True
		for s in range(env.nS):
			old_a = policy[s]
			# update the policy with the action that bring to the highest state value
			policy[s] = np.argmax([eval_state_action(V, s, a, env=env) for a in range(env.nA)])
			if old_a != policy[s]:
				policy_stable = False

		return policy_stable
	except Exception as policy_improvement_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'policy_improvement'", policy_improvement_exception)


def run_policy_iteration(*kwargs):
	try:
		temp_kwargs = copy.deepcopy(kwargs)
		temp_kwargs[0]["env"].reset()
		print(f"\nStarting: Policy Iteration\n")
		env_evaluate_copy = copy.deepcopy(temp_kwargs[0]["env"])

		temp_results = get_time_wrapper(policy_iteration, *temp_kwargs)
		avg_scores, scores = evaluate_policy(env=env_evaluate_copy, policy=temp_results["Results"]["Policy"],
											 gamma=temp_kwargs[0]["gamma"])
		temp_results["Results"]["Policy_Scores"] = scores
		temp_results["Results"]["Average_Scores"] = np.mean(scores)
		temp_results["Results"]["Highest_Score"] = np.max(temp_results['Results']['Total_Value'])
		print(f"Policy Iteration"
			  f"\n\tNumber of States: {temp_kwargs[0]['env'].nS}"
			  f"\n\tMax Iterations: {temp_kwargs[0]['max_iterations']}"
			  f"\n\tGamma: {temp_kwargs[0]['gamma']}"
			  f"\n\tRun Time: {temp_results['Elapsed Time']:.5f}s"
			  f"\n\tAverage Score: {avg_scores:.3f}"
			  f"\n\tHighest Score: {np.max(temp_results['Results']['Total_Value']):.3f}")

		return temp_results
	except Exception as run_policy_iteration_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'run_policy_iteration'", run_policy_iteration_exception)
