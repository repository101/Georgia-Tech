import os
import sys
import time

import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from gym.envs.toy_text.frozen_lake import generate_random_map
from hiive import mdptoolbox
from matplotlib import gridspec

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500

plt.tight_layout()

save_dir = "Graphs"
mpl.rcParams['agg.path.chunksize'] = 10000


def get_time_wrapper(some_function, kwargs):
	try:
		start_time = time.perf_counter()
		result = some_function(**kwargs)
		end_time = time.perf_counter()
		elapsed_time = end_time - start_time
		return {"Elapsed Time": elapsed_time, "Results": result}
	except Exception as get_time_wrapper_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'get_time_wrapper'", get_time_wrapper_exception)


def extract_data_to_frame(run_stats, max_iterations, ending_iteration=0, increase_to_max=False, is_forest=False,
						  is_q_learner=False, normalize=True):
	try:
		if is_forest:
			if increase_to_max:
				temp_df = pd.DataFrame.from_dict(run_stats)
				result_df = pd.DataFrame(index=np.arange(max_iterations), columns=temp_df.columns)
				result_df.loc[:, :] = temp_df
				# result_df["Iteration"] = pd.to_numeric(result_df["Iteration"], errors="coerce")
				result_df.loc[0, ["Reward", "Error", "Time", "Max V", "Mean V"]] += 0.00001
				result_df.fillna(method="ffill", inplace=True)
				old_time = result_df["Time"].copy()
				old_reward = result_df["Reward"].copy()
				if normalize:
					result_df /= result_df.iloc[0]
				result_df["Time"] = old_time
				result_df["Reward"] = old_reward
				result_df["Reward_Pct_Change"] = result_df["Reward"].pct_change(periods=1)
				result_df["Reward_Cumulative_Sum"] = result_df["Reward"].cumsum()
				result_df["Error_Pct_Change"] = result_df["Error"].pct_change(periods=1)
				result_df["Time_Pct_Change"] = result_df["Time"].pct_change(periods=1)
				result_df["Time_Per_Iteration"] = result_df["Time"].diff()
				result_df["Total_Time"] = result_df["Time"].diff().cumsum()
				return result_df.astype("float")
			else:
				temp_df = pd.DataFrame.from_dict(run_stats)
				result_df = pd.DataFrame.from_dict(run_stats)
				result_df.loc[0, ["Reward", "Error", "Time", "Max V", "Mean V"]] += 0.00001
				if normalize:
					result_df /= result_df.iloc[0]
				result_df["Reward_Pct_Change"] = result_df["Reward"].pct_change(periods=1)
				result_df["Reward_Cumulative_Sum"] = result_df["Reward"].cumsum()
				result_df["Error_Pct_Change"] = result_df["Error"].pct_change(periods=1)
				result_df["Time_Pct_Change"] = result_df["Time"].pct_change(periods=1)
				result_df["Total_Time"] = result_df["Time"].cumsum()
				return result_df.astype("float")
		else:
			result_df = pd.DataFrame.from_dict(run_stats)
			old_df = result_df.copy()
			if increase_to_max:
				a = result_df.loc[5:, :]
				a[a == 0] = np.nan
			result_df = result_df.loc[0:ending_iteration, :]
			result_df.fillna(method="ffill", inplace=True)
			if is_q_learner:
				result_df["Reward_Pct_Change"] = result_df["Rewards_Per_Iteration"].pct_change(periods=1)
				result_df["Total_Rewards"] = result_df["Rewards_Per_Iteration"].cumsum()
				result_df["Time_Pct_Change"] = result_df["Run_Times"].pct_change(periods=1)
				result_df["Total_Time"] = result_df["Run_Times"].cumsum()
			else:
				result_df["Reward_Pct_Change"] = result_df["Total_Value"].pct_change(periods=1)
				result_df["Time_Pct_Change"] = result_df["Run_Times"].pct_change(periods=1)
				result_df["Total_Time"] = result_df["Run_Times"].cumsum()
			return result_df.astype("float")
	except Exception as extract_data_to_frame_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'extract_data_to_frame'", extract_data_to_frame_exception)


def generate_frozen_lake(size="s", map_size=10, p=0.8, seed=42, set_specific_size=False, is_slippery=False):
	try:
		if isinstance(size, str):
			if size.lower() == "s":
				random_map = generate_random_map(size=4, p=p)
				env = gym.make("FrozenLake-v0", desc=random_map, is_slippery=is_slippery)
				return env
			elif size.lower() == "l" or size.lower() == "b":
				random_map = generate_random_map(size=10, p=p)
				env = gym.make("FrozenLake-v0", desc=random_map, is_slippery=is_slippery)
				return env
		else:
			if set_specific_size:
				random_map = generate_random_map(size=map_size, p=p)
				env = gym.make("FrozenLake-v0", desc=random_map, is_slippery=is_slippery)
				return env
			print(f"Incorrect size specified, passed in value {size}")
	except Exception as generate_frozen_lake_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'generate_frozen_lake'", generate_frozen_lake_exception)


def generate_forest_mgmt(size="s", seed=42, set_specific_size=False):
	try:
		if isinstance(size, str):
			if size.lower() == "s":
				P, R = mdptoolbox.forest(S=4)
				return P, R
			elif size.lower() == "l" or size.lower() == "b":
				P, R = mdptoolbox.forest(S=10)
				return P, R
			else:
				print(f"Incorrect size specified, passed in value {size}")
				return
		else:
			if set_specific_size:
				P, R = mdptoolbox.forest(S=size, p=0)
				return P, R
			print(f"Incorrect size specified, passed in value {size}")
	except Exception as generate_forest_mgmt_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'generate_forest_mgmt'", generate_forest_mgmt_exception)


def extract_policy(env, value_function, gamma=1.0):
	""" Extract the policy for a given value function

	https://learning.oreilly.com/library/view/deep-learning-with/9781838553005/27f144fb-4839-4d9b-ba9e-023191b9fd12.xhtml
	"""
	try:
		policy = np.zeros(env.nS)
		for state in range(env.nS):
			q_sa = np.zeros(env.nA)
			for action in range(env.nA):
				q_sa[action] = sum([policy * (_reward + gamma * value_function[_state]) for
									policy, _state, _reward, _ in env.P[state][action]])
			policy[state] = np.argmax(q_sa)
		return policy
	except Exception as extract_policy_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'extract_policy'", extract_policy_exception)


def run_episode(env, policy, gamma=1.0, max_attempts=100):
	""" Runs an episode and return the total reward
	https://learning.oreilly.com/library/view/deep-learning-with/9781838553005/27f144fb-4839-4d9b-ba9e-023191b9fd12.xhtml
	"""
	try:
		obs = env.reset()
		total_reward = 0
		step_index = 0
		for i in range(max_attempts):
			obs, reward, done, _ = env.step(int(policy[obs]))
			total_reward += (gamma ** step_index * reward)
			step_index += 1
			if done:
				break
		return total_reward
	except Exception as run_episode_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'run_episode'", run_episode_exception)


def evaluate_policy(env, policy, gamma=1.0, n=1000):
	"""
	:param env:
	:param policy:
	:param gamma:
	:param n:
	:return:
	https://learning.oreilly.com/library/view/deep-learning-with/9781838553005/27f144fb-4839-4d9b-ba9e-023191b9fd12.xhtml
	"""
	try:
		scores = [
			run_episode(env, policy, gamma=gamma)
			for _ in range(n)]
		return np.mean(scores), scores
	except Exception as evaluate_policy_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'evaluate_policy'", evaluate_policy_exception)


def check_folder(_directory):
	try:
		MYDIR = os.getcwd() + "\\" + _directory
		CHECK_FOLDER = os.path.isdir(MYDIR)

		# If folder doesn't exist, then create it.
		if not CHECK_FOLDER:
			os.makedirs(MYDIR)
			print("created folder : ", MYDIR)
		else:
			print(MYDIR, "folder already exists.")
	except Exception as check_folder_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'check_folder'", check_folder_exception)


def plot_iteration_vs_reward(dataframe, title, xlabel, ylabel, legend_label="", filename="", is_frozen=False,
							 is_frozen_q=False, is_forest=False, plot_runtime=False):
	try:
		plt.close("all")
		plt.style.use("ggplot")
		fig, ax1 = plt.subplots(figsize=(12, 8))
		step = 1

		if is_frozen:
			ln1 = ax1.plot(dataframe["Total_Value"], label=f"{legend_label} Reward", color="darkorange")
		elif is_frozen_q:
			ln1 = ax1.plot(dataframe["Rewards_Per_Iteration"], label=f"{legend_label} Reward", color="darkorange")
		if is_forest:
			cut_off_point = dataframe["Iteration"].idxmax()
			if cut_off_point > 1000:
				step = 3
			ln1 = ax1.plot(dataframe["Reward"].iloc[0:cut_off_point:step], label=f"{legend_label} Reward",
						   color="darkorange")

		if plot_runtime:
			ax1_1 = ax1.twinx()
			ax1.tick_params(axis="y", labelcolor="darkorange")
			if is_forest:
				cut_off_point = dataframe["Iteration"].idxmax()
				if cut_off_point > 1000:
					step = 3
				ln2 = ax1_1.plot(dataframe["Time"].iloc[0:cut_off_point:step], label=f"{legend_label} Runtime",
								 color="navy")
			else:
				ln2 = ax1_1.plot(dataframe["Run_Times"], label=f"{legend_label} Runtime", color="navy")
			ax1_1.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.01)
			ax1_1.set_ylabel(f"Runtime", fontsize=15, weight='heavy')
			ax1_1.tick_params(axis="y", labelcolor="navy")
			plt.setp(ax1_1.get_yticklabels(), rotation=-30, ha="left", rotation_mode="anchor")

		ax1.set_title(f"{title}", fontsize=15, weight='bold')
		ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
		ax1.set_xlabel(f"{xlabel}", fontsize=15, weight='heavy')
		ax1.set_ylabel(f"{ylabel}", fontsize=15, weight='heavy')

		if plot_runtime:
			lns = ln1 + ln2
		else:
			lns = ln1

		labs = [l.get_label() for l in lns]
		if plot_runtime:
			ax1_1.legend(lns, labs, loc="best", markerscale=1.1, frameon=True,
						 edgecolor="black", fancybox=True, shadow=True)
			plt.savefig(f"{os.getcwd()}/{save_dir}/{filename}_Iteration_Vs_Reward_Vs_Runtime.png")
		else:
			ax1.legend(lns, labs, loc="best", markerscale=1.1, frameon=True,
					   edgecolor="black", fancybox=True, shadow=True)
			plt.savefig(f"{os.getcwd()}/{save_dir}/{filename}_Iteration_Vs_Reward.png")

		return
	except Exception as plot_iteration_vs_reward_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'plot_iteration_vs_reward'", plot_iteration_vs_reward_exception)


def plot_combined_iteration_vs_reward_frozen(value_iteration_df, policy_iteration_df, q_learner_df,
											 title, xlabel, ylabel, filename="", max_iterations=10000):
	try:
		step = 1
		value_cut_off = value_iteration_df["Run_Times"].idxmin()
		policy_cut_off = policy_iteration_df["Run_Times"].idxmin()
		combined_cut_off = max(value_cut_off, policy_cut_off)
		if combined_cut_off > 1000:
			step = 3

		plt.close("all")
		plt.style.use("ggplot")
		fig = plt.figure(figsize=(16, 10), constrained_layout=True)
		gs = gridspec.GridSpec(3, 1, height_ratios=[4, 4, 4], figure=fig, wspace=0.05, hspace=0.15)
		ax0 = plt.subplot(gs[0])
		ax1 = plt.subplot(gs[1])
		ax2 = plt.subplot(gs[2])

		ax0.set_title(f"{title}", fontsize=20, weight='bold')

		ax0.plot(value_iteration_df["Run_Times"].iloc[0:combined_cut_off], label="Value Iteration",
				 linewidth=3, color="navy")
		ax1.plot(policy_iteration_df["Run_Times"].iloc[0:combined_cut_off],
				 label="Policy Iteration", linewidth=3, color="navy")
		ax2.plot(q_learner_df["Run_Times"].iloc[0::step], label="Q Learner", linewidth=3, color="navy")

		ax0.tick_params(which="minor", bottom=False, left=False, labelsize=15)
		ax0.tick_params(which="major", bottom=True, left=True, labelsize=15)
		plt.setp(ax0.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
		ax0.set_ylabel(f"{ylabel}", fontsize=10, weight='heavy')
		ax0.grid(which='both', linestyle='-', linewidth='0.5', color='white')
		ax0.legend(loc="best", markerscale=1.1, frameon=True,
				   edgecolor="black", fancybox=True, shadow=True, fontsize=12)

		ax1.tick_params(which="minor", bottom=False, left=False, labelsize=15)
		ax1.tick_params(which="major", bottom=True, left=True, labelsize=15)
		plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
		ax1.set_ylabel(f"{ylabel}", fontsize=10, weight='heavy')
		ax1.grid(which='both', linestyle='-', linewidth='0.5', color='white')
		ax1.legend(loc="best", markerscale=1.1, frameon=True,
				   edgecolor="black", fancybox=True, shadow=True, fontsize=12)

		ax2.tick_params(which="minor", bottom=False, left=False, labelsize=15)
		ax2.tick_params(which="major", bottom=True, left=True, labelsize=15)
		plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
		ax2.set_ylabel(f"{ylabel}", fontsize=10, weight='heavy')
		ax2.set_xlabel(f"{xlabel}", fontsize=20, weight='heavy')
		ax2.grid(which='both', linestyle='-', linewidth='0.5', color='white')
		ax2.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True,
				   shadow=True, fontsize=12)
		plt.savefig(f"{os.getcwd()}/{save_dir}/{filename}_Iteration_Vs_Reward_Combined.png")
		return
	except Exception as plot_iteration_vs_reward_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'plot_iteration_vs_reward'", plot_iteration_vs_reward_exception)


def plot_iteration_vs_runtime(dataframe, title, xlabel, ylabel, legend_label="", filename="", is_frozen=False,
							  is_frozen_q=False):
	try:
		plt.close("all")
		plt.style.use("ggplot")
		fig, ax1 = plt.subplots(figsize=(12, 8))
		step = 1
		if is_frozen:
			ax1.plot(dataframe["Run_Times"], label=f"{legend_label} Runtime", color="darkorange")
		elif is_frozen_q:
			ax1.plot(dataframe["Run_Times"], label=f"{legend_label} Runtime", color="darkorange")
		else:
			cut_off_point = dataframe["Iteration"].idxmax()
			if cut_off_point > 1000:
				step = 3
			ax1.plot(dataframe["Run_Times"].iloc[0:cut_off_point:step], label=f"{legend_label} Runtime",
					 color="darkorange")

		ax1.set_title(f"{title}", fontsize=15, weight='bold')
		ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
		ax1.set_xlabel(f"{xlabel}", fontsize=15, weight='heavy')
		ax1.set_ylabel(f"{ylabel}", fontsize=15, weight='heavy')
		ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True, shadow=True)
		plt.savefig(f"{os.getcwd()}/{save_dir}/{filename}_Iteration_Vs_Runtime.png")
		return
	except Exception as plot_iteration_vs_reward_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'plot_iteration_vs_reward'", plot_iteration_vs_reward_exception)


def plot_iteration_vs_error(dataframe, title, xlabel, ylabel, legend_label="", filename="", plot_runtime=False):
	try:
		plt.close("all")
		plt.style.use("ggplot")
		fig, ax1 = plt.subplots(figsize=(12, 8))
		cut_off_point = dataframe["Iteration"].idxmax()
		step = 1
		if cut_off_point > 1000:
			step = 3
		ln1 = ax1.plot(dataframe["Error"].iloc[0:cut_off_point:step], label=f"{legend_label} Error", color="darkorange")

		if plot_runtime:
			ax1_1 = ax1.twinx()
			ax1.tick_params(axis="y", labelcolor="darkorange")
			ln2 = ax1_1.plot(dataframe["Time"].iloc[0:cut_off_point:step], label=f"{legend_label} Runtime",
							 color="navy")
			ax1_1.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.01)
			ax1_1.set_ylabel(f"Runtime", fontsize=15, weight='heavy')
			ax1_1.tick_params(axis="y", labelcolor="navy")
			plt.setp(ax1_1.get_yticklabels(), rotation=-30, ha="left", rotation_mode="anchor")

		ax1.set_title(f"{title}", fontsize=15, weight='bold')
		ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
		ax1.set_xlabel(f"{xlabel}", fontsize=15, weight='heavy')
		ax1.set_ylabel(f"{ylabel}", fontsize=15, weight='heavy')

		if plot_runtime:
			lns = ln1 + ln2
		else:
			lns = ln1

		labs = [l.get_label() for l in lns]
		if plot_runtime:
			ax1_1.legend(lns, labs, loc="best", markerscale=1.1, frameon=True,
						 edgecolor="black", fancybox=True, shadow=True)
			plt.savefig(f"{os.getcwd()}/{save_dir}/{filename}_Iteration_Vs_Error_Vs_Runtime.png")
		else:
			ax1.legend(lns, labs, loc="best", markerscale=1.1, frameon=True,
					   edgecolor="black", fancybox=True, shadow=True)
			plt.savefig(f"{os.getcwd()}/{save_dir}/{filename}_Iteration_Vs_Error.png")
		return
	except Exception as plot_iteration_vs_reward_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'plot_iteration_vs_reward'", plot_iteration_vs_reward_exception)


def plot_combined_iteration_vs_runtime_frozen(value_iteration_df, policy_iteration_df, q_learner_df,
											  title, xlabel, ylabel, filename="", max_iterations=10000):
	try:
		step = 1
		value_cut_off = value_iteration_df["Run_Times"].idxmin()
		policy_cut_off = policy_iteration_df["Run_Times"].idxmin()
		combined_cut_off = max(value_cut_off, policy_cut_off)
		if combined_cut_off > 1000:
			step = 3
		plt.close("all")
		plt.style.use("ggplot")

		fig = plt.figure(figsize=(16, 10), constrained_layout=True)
		gs = gridspec.GridSpec(3, 1, height_ratios=[4, 4, 4], figure=fig, wspace=0.05, hspace=0.15)
		ax0 = plt.subplot(gs[0])
		ax1 = plt.subplot(gs[1])
		ax2 = plt.subplot(gs[2])

		ax0.set_title(f"{title}", fontsize=20, weight='bold')

		ax0.plot(value_iteration_df["Run_Times"].iloc[0:combined_cut_off], label="Value Iteration",
				 linewidth=3, color="navy")
		ax1.plot(policy_iteration_df["Run_Times"].iloc[0:combined_cut_off],
				 label="Policy Iteration", linewidth=3, color="navy")
		ax2.plot(q_learner_df["Run_Times"].iloc[0::step], label="Q Learner", linewidth=3, color="navy")

		ax0.tick_params(which="minor", bottom=False, left=False, labelsize=15)
		ax0.tick_params(which="major", bottom=True, left=True, labelsize=15)
		plt.setp(ax0.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
		ax0.set_ylabel(f"{ylabel}", fontsize=15, weight='heavy')
		ax0.grid(which='both', linestyle='-', linewidth='0.5', color='white')
		ax0.legend(loc="best", markerscale=1.1, frameon=True,
				   edgecolor="black", fancybox=True, shadow=True, fontsize=12)

		ax1.tick_params(which="minor", bottom=False, left=False, labelsize=15)
		ax1.tick_params(which="major", bottom=True, left=True, labelsize=15)
		plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
		ax1.set_ylabel(f"{ylabel}", fontsize=15, weight='heavy')
		ax1.grid(which='both', linestyle='-', linewidth='0.5', color='white')
		ax1.legend(loc="best", markerscale=1.1, frameon=True,
				   edgecolor="black", fancybox=True, shadow=True, fontsize=12)

		ax2.tick_params(which="minor", bottom=False, left=False, labelsize=15)
		ax2.tick_params(which="major", bottom=True, left=True, labelsize=15)
		plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
		ax2.set_ylabel(f"{ylabel}", fontsize=15, weight='heavy')
		ax2.set_xlabel(f"{xlabel}", fontsize=20, weight='heavy')
		ax2.grid(which='both', linestyle='-', linewidth='0.5', color='white')
		ax2.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True,
				   shadow=True, fontsize=12)
		plt.savefig(f"{os.getcwd()}/{save_dir}/{filename}_Iteration_Vs_Runtime_Combined.png")
		return
	except Exception as plot_iteration_vs_reward_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'plot_iteration_vs_reward'", plot_iteration_vs_reward_exception)


def plot_combined_iteration_vs_runtime_forest(value_iteration_df, policy_iteration_df, q_learner_df,
											  title, xlabel, ylabel, filename="", max_iterations=10000):
	try:
		plt.close("all")
		plt.style.use("ggplot")
		fig = plt.figure(figsize=(16, 10), constrained_layout=True)
		gs = gridspec.GridSpec(3, 1, height_ratios=[4, 4, 4], figure=fig, wspace=0.05, hspace=0.15)
		ax0 = plt.subplot(gs[0])
		ax1 = plt.subplot(gs[1])
		ax2 = plt.subplot(gs[2])
		step = 1
		value_cut_off = value_iteration_df["Iteration"].idxmax()
		policy_cut_off = policy_iteration_df["Iteration"].idxmax()
		combined_cut_off = max(value_cut_off, policy_cut_off)
		if combined_cut_off > 1000:
			step = 3

		ax0.set_title(f"{title}", fontsize=20, weight='bold')
		ax0.plot(value_iteration_df["Time"].iloc[0:combined_cut_off:step], label="Value Iteration",
				 linewidth=3, color="navy")
		ax1.plot(policy_iteration_df["Time"].iloc[0:combined_cut_off:step],
				 label="Policy Iteration", linewidth=3, color="navy")
		ax2.plot(q_learner_df["Time"].iloc[0::step], label="Q Learner", linewidth=3, color="navy")

		ax0.tick_params(which="minor", bottom=False, left=False, labelsize=15)
		ax0.tick_params(which="major", bottom=True, left=True, labelsize=15)
		plt.setp(ax0.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
		ax0.set_ylabel(f"{ylabel}", fontsize=15, weight='heavy')
		ax0.grid(which='both', linestyle='-', linewidth='0.5', color='white')
		ax0.legend(loc="best", markerscale=1.1, frameon=True,
				   edgecolor="black", fancybox=True, shadow=True, fontsize=12)

		ax1.tick_params(which="minor", bottom=False, left=False, labelsize=15)
		ax1.tick_params(which="major", bottom=True, left=True, labelsize=15)
		plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
		ax1.set_ylabel(f"{ylabel}", fontsize=15, weight='heavy')
		ax1.grid(which='both', linestyle='-', linewidth='0.5', color='white')
		ax1.legend(loc="best", markerscale=1.1, frameon=True,
				   edgecolor="black", fancybox=True, shadow=True, fontsize=12)

		ax2.tick_params(which="minor", bottom=False, left=False, labelsize=15)
		ax2.tick_params(which="major", bottom=True, left=True, labelsize=15)
		plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
		ax2.set_ylabel(f"{ylabel}", fontsize=15, weight='heavy')
		ax2.set_xlabel(f"{xlabel}", fontsize=20, weight='heavy')
		ax2.grid(which='both', linestyle='-', linewidth='0.5', color='white')
		ax2.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True,
				   shadow=True, fontsize=12)

		plt.savefig(f"{os.getcwd()}/{save_dir}/{filename}_Iteration_Vs_Runtime_Combined.png")
		return
	except Exception as plot_iteration_vs_reward_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'plot_iteration_vs_reward'", plot_iteration_vs_reward_exception)


def plot_combined_iteration_vs_error_forest(value_iteration_df, policy_iteration_df, q_learner_df,
											title, xlabel, ylabel, filename="", max_iterations=10000):
	try:
		step = 1
		value_cut_off = value_iteration_df["Iteration"].idxmax()
		policy_cut_off = policy_iteration_df["Iteration"].idxmax()
		combined_cut_off = max(value_cut_off, policy_cut_off)
		if combined_cut_off > 1000:
			step = 3
		plt.close("all")
		plt.style.use("ggplot")
		fig = plt.figure(figsize=(16, 10), constrained_layout=True)
		gs = gridspec.GridSpec(3, 1, height_ratios=[4, 4, 4], figure=fig, wspace=0.05, hspace=0.15)
		ax0 = plt.subplot(gs[0])
		ax1 = plt.subplot(gs[1])
		ax2 = plt.subplot(gs[2])

		ax0.set_title(f"{title}", fontsize=20, weight='bold')

		ax0.plot(value_iteration_df["Error"].iloc[0:combined_cut_off:step], label="Value Iteration",
				 linewidth=3, color="navy")
		ax1.plot(policy_iteration_df["Error"].iloc[0:combined_cut_off:step],
				 label="Policy Iteration", linewidth=3, color="navy")
		ax2.plot(q_learner_df["Error"].iloc[0::step], label="Q Learner", linewidth=3, color="navy")

		ax0.tick_params(which="minor", bottom=False, left=False, labelsize=15)
		ax0.tick_params(which="major", bottom=True, left=True, labelsize=15)
		plt.setp(ax0.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
		ax0.set_ylabel(f"{ylabel}", fontsize=10, weight='heavy')
		ax0.grid(which='both', linestyle='-', linewidth='0.5', color='white')
		ax0.legend(loc="best", markerscale=1.1, frameon=True,
				   edgecolor="black", fancybox=True, shadow=True, fontsize=12)

		ax1.tick_params(which="minor", bottom=False, left=False, labelsize=15)
		ax1.tick_params(which="major", bottom=True, left=True, labelsize=15)
		plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
		ax1.set_ylabel(f"{ylabel}", fontsize=10, weight='heavy')
		ax1.grid(which='both', linestyle='-', linewidth='0.5', color='white')
		ax1.legend(loc="best", markerscale=1.1, frameon=True,
				   edgecolor="black", fancybox=True, shadow=True, fontsize=12)

		ax2.tick_params(which="minor", bottom=False, left=False, labelsize=15)
		ax2.tick_params(which="major", bottom=True, left=True, labelsize=15)
		plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
		ax2.set_ylabel(f"{ylabel}", fontsize=10, weight='heavy')
		ax2.set_xlabel(f"{xlabel}", fontsize=20, weight='heavy')
		ax2.grid(which='both', linestyle='-', linewidth='0.5', color='white')
		ax2.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True,
				   shadow=True, fontsize=12)

		plt.savefig(f"{os.getcwd()}/{save_dir}/{filename}_Iteration_Vs_Error_Combined.png")
		return
	except Exception as plot_iteration_vs_reward_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'plot_iteration_vs_reward'", plot_iteration_vs_reward_exception)


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", valfmt="{x:.2f}",
			textcolors=["white", "black"], threshold=None, x_label=None, y_label=None, title=None,
			filename="", folder=None, cmap="viridis", annotate=False, title_size=15, axis_size=15,
			cbar_fontsize=15, set_label_tick_marks=False, is_forest=False, no_colorbar=False, **kwargs):
	"""
	https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

	Create a heatmap from a numpy array and two lists of labels.

	Parameters
	----------
	data
		A 2D numpy array of shape (N, M).
	row_labels
		A list or array of length N with the labels for the rows.
	col_labels
		A list or array of length M with the labels for the columns.
	ax
		A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
		not provided, use current axes or create a new one.  Optional.
	cbar_kw
		A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
	cbarlabel
		The label for the colorbar.  Optional.
	**kwargs
		All other arguments are forwarded to `imshow`.
	"""
	try:
		print("Starting Heatmap")
		if not ax:
			ax = plt.gca()

		plt.style.use("ggplot")
		# Plot the heatmap
		if title is not None:
			ax.set_title(title, fontsize=title_size, weight='bold')

		im = ax.imshow(data, cmap=cmap)
		if x_label is not None:
			ax.set_xlabel(x_label, fontsize=axis_size, weight='heavy')

		if y_label is not None:
			ax.set_ylabel(y_label, fontsize=axis_size, weight='heavy')

		# Create colorbar
		if not no_colorbar:
			cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
			cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", weight="heavy", fontsize=cbar_fontsize)

		# We want to show all ticks...
		ax.set_xticks(np.arange(data.shape[1]))
		ax.set_yticks(np.arange(data.shape[0]))
		# ... and label them with the respective list entries.
		if set_label_tick_marks:
			ax.set_xticklabels(col_labels)
			ax.set_yticklabels(row_labels)

		# Let the horizontal axes labeling appear on top.
		ax.tick_params(top=False, bottom=True,
					   labeltop=False, labelbottom=True)

		# Rotate the tick labels and set their alignment.
		plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
				 rotation_mode="anchor")

		# Turn spines off and create white grid.
		for edge, spine in ax.spines.items():
			spine.set_visible(False)

		ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
		ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
		ax.grid(which="minor", color="w", linestyle='-', linewidth='1', alpha=0.5)
		ax.grid(which='major', linestyle='--', linewidth='0', color='white', alpha=0)
		ax.tick_params(which="minor", bottom=False, left=False)
		if annotate:
			final_heatmap = annotate_heatmap(im=im, valfmt=valfmt, textcolors=textcolors, threshold=threshold,
											 is_forest=is_forest)
		# plt.tight_layout()
		print("Heatmap Finished")
		return ax
	except Exception as heatmap_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'heatmap'", heatmap_exception)


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"], is_forest=False,
					 threshold=None, **textkw):
	"""
	https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
	A function to annotate a heatmap.

	Parameters
	----------
	im
		The AxesImage to be labeled.
	data
		Data used to annotate.  If None, the image's data is used.  Optional.
	valfmt
		The format of the annotations inside the heatmap.  This should either
		use the string format method, e.g. "$ {x:.2f}", or be a
		`matplotlib.ticker.Formatter`.  Optional.
	textcolors
		A list or array of two color specifications.  The first is used for
		values below a threshold, the second for those above.  Optional.
	threshold
		Value in data units according to which the colors from textcolors are
		applied.  If None (the default) uses the middle of the colormap as
		separation.  Optional.
	**kwargs
		All other arguments are forwarded to each call to `text` used to create
		the text labels.
	"""
	try:
		if is_forest:
			label = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
		else:
			label = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
		if not isinstance(data, (list, np.ndarray)):
			data = im.get_array()

		# Normalize the threshold to the images color range.
		if threshold is not None:
			threshold = im.norm(threshold)
		else:
			threshold = im.norm(data.max()) / 2.

		# Set default alignment to center, but allow it to be
		# overwritten by textkw.
		kw = dict(horizontalalignment="center",
				  verticalalignment="center")
		kw.update(textkw)

		# Get the formatter in case a string is supplied
		if isinstance(valfmt, str):
			valfmt = mpl.ticker.StrMethodFormatter(valfmt)

		# Loop over the data and create a `Text` for each "pixel".
		# Change the text's color depending on the data.
		texts = []
		for i in range(data.shape[0]):
			for j in range(data.shape[1]):
				kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
				if i == 0 and j == 0:
					text = im.axes.text(j, i, "Start", **kw)
				elif i == 9 and j == 9:
					text = im.axes.text(j, i, "Finish", **kw)
				else:
					text = im.axes.text(j, i, label[data[i, j]], **kw)
				texts.append(text)

		return texts
	except Exception as annotate_heatmap_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'annotate_heatmap'", annotate_heatmap_exception)


def load_pickles():
	try:
		try:
			all_dataframes = {}
			for sz in ["small", "large"]:
				for alg in ["policy_iteration", "value_iteration", "q_learner"]:
					for param in ["gamma", "eps"]:
						for prob in ["forest", "frozen"]:
							for result_type in ["reward", "runtime"]:
								name = f"{sz}_{alg}_{param}_{prob}_{result_type}"
								with open(f"{os.getcwd()}/{name}.pkl", "rb") as input_file:
									all_dataframes[name] = pickle.load(input_file)
									input_file.close()
			return all_dataframes
		except Exception as _exception:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)
			print(f"Exception in ''", _exception)
	except Exception as _exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in ''", _exception)
