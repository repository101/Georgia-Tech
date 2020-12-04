import copy
import os
import sys
import time

import numpy as np

from util import get_time_wrapper


def eps_greedy(Q, s, eps=0.1):
	"""
	https://github.com/PacktPublishing/Reinforcement-Learning-Algorithms-with-Python/blob/master/Chapter04/SARSA%20Q_learning%20Taxi-v2.py
	Epsilon greedy policy
	"""
	try:
		if np.random.uniform(0, 1) < eps:
			# Choose a random action
			return np.random.randint(Q.shape[1])
		else:
			# Choose the action of a greedy policy
			return greedy(Q, s)
	except Exception as eps_greedy_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'eps_greedy'", eps_greedy_exception)


def greedy(Q, s):
	"""
	Greedy policy
	return the index corresponding to the maximum action-state value
	"""
	try:
		return np.argmax(Q[s])
	except Exception as greedy_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'greedy'", greedy_exception)


def run_episodes(env, Q, num_episodes=100, max_attempts=100, eval=False):
	"""
	https://github.com/PacktPublishing/Reinforcement-Learning-Algorithms-with-Python/blob/master/Chapter04/SARSA%20Q_learning%20Taxi-v2.py
	Run some episodes to test the policy
	"""
	try:
		tot_rew = []
		state = 0
		if not eval:
			state = env.reset()
		for _ in range(num_episodes):
			done = False
			game_rew = 0
			for _ in range(max_attempts):
				# select a greedy action
				next_state, rew, done, _ = env.step(greedy(Q, state))
				state = next_state
				game_rew += rew
				if done:
					state = env.reset()
					tot_rew.append(game_rew)
					break
		if len(tot_rew) > 0:
			return np.mean(tot_rew), tot_rew
		else:
			return 0, 0
	except Exception as run_episodes_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'run_episodes'", run_episodes_exception)


def Q_learning(env, learning_rate=0.9, max_iterations=5000, eps=0.001, gamma=0.95, eps_decay=1e-5, is_large=False):
	"""
	https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
	:param env:
	:param learning_rate:
	:param max_iterations:
	:param eps:
	:param gamma:
	:param eps_decay:
	:return:
	"""
	try:
		# Initialize table with all zeros
		env_copy = copy.deepcopy(env)
		Max_reward = 0
		num_reset = 0
		converge_limit = 100
		converge_check_limit = 100
		reset_limit = 5
		if is_large:
			converge_limit = 8000
			converge_check_limit = 4000
			reset_limit = 10
		step_size = 1 / max_iterations
		Q = np.zeros([env.observation_space.n, env.action_space.n])
		run_times = np.zeros(shape=(max_iterations,))
		steps_per_episode = np.zeros(shape=(max_iterations,))
		reward_array = np.zeros(shape=(max_iterations,))
		iteration = 0
		converge_count = 0
		old_eps = copy.copy(eps)
		for iteration in range(max_iterations):
			start_time = time.perf_counter()
			current_state = env.reset()
			rAll = 0
			is_done = False
			num_steps = 0
			while num_steps < 200:
				num_steps += 1
				if np.random.uniform(0, 1) > eps:
					current_action = env.action_space.sample()
				else:
					current_action = max(env.P[current_state], key=env.P[current_state].get)
				next_state, reward, is_done, info = env.step(current_action)
				Q[current_state, current_action] = Q[current_state, current_action] + \
												   learning_rate * (reward + gamma * np.max(Q[next_state, :]) -
																	Q[current_state, current_action])
				rAll += reward
				current_state = next_state
				if is_done:
					break
			if eps < 1.0:
				eps += eps_decay
			# eps += (eps_decay * (1 - (step_size * iteration)))
			end_time = time.perf_counter()
			run_times[iteration] = end_time - start_time
			steps_per_episode[iteration] = num_steps
			reward_array[iteration] = np.sum(np.max(Q, axis=1))
			Max_reward = np.max(reward_array)
			if iteration > converge_check_limit and np.round(reward_array[iteration], 5) != 0 \
					and np.round(reward_array[iteration], 5) == np.round(reward_array[iteration - 1], 5):
				converge_count += 1
			else:
				converge_count = 0
			if converge_count >= converge_limit:
				num_reset += 1
				if num_reset > reset_limit:
					break
				else:
					# Reset epsilon
					converge_count = 0
					eps = old_eps
		return {"Q_Table": Q, "Steps_Per_Iteration": steps_per_episode, "Run_Times": run_times,
				"Rewards_Per_Iteration": reward_array, "Average_Run_Time": np.mean(run_times[:iteration]),
				"Number_Of_Iterations": iteration, "Optimal_Policy": np.argmax(Q, axis=1)}
	except Exception as Q_learning_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'Q_learning'", Q_learning_exception)


def run_q_learner(*kwargs, is_large=False, test=False):
	try:
		temp_kwargs = copy.deepcopy(kwargs)
		temp_kwargs[0]["env"].reset()
		temp_kwargs[0]["is_large"] = is_large
		if is_large:
			if not test:
				temp_kwargs[0]["eps"] = 0.00001
				temp_kwargs[0]["eps_decay"] = 9e-06
				temp_kwargs[0]["max_iterations"] = 400000
				temp_kwargs[0]["learning_rate"] = 0.9
		print(f"\nStarting: Q Learning")
		temp_results = get_time_wrapper(Q_learning, *temp_kwargs)
		avg_reward, all_episode_rewards = run_episodes(env=temp_kwargs[0]["env"], Q=temp_results["Results"]["Q_Table"],
													   num_episodes=1000)
		temp_results["Results"]["Policy_Scores"] = all_episode_rewards
		temp_results["Results"]["Average_Scores"] = avg_reward
		print(f"Q Learner"
			  f"\n\tNumber of States: {temp_kwargs[0]['env'].nS}"
			  f"\n\tMax Iterations: {temp_kwargs[0]['max_iterations']}"
			  f"\n\tGamma: {temp_kwargs[0]['gamma']}"
			  f"\n\tRun Time: {temp_results['Elapsed Time']:.5f}s"
			  f"\n\tAverage Score: {avg_reward:.3f}"
			  f"\n\tHighest Score: {np.max(temp_results['Results']['Rewards_Per_Iteration']):.3f}")
		return temp_results
	except Exception as run_policy_iteration_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'run_policy_iteration'", run_policy_iteration_exception)
