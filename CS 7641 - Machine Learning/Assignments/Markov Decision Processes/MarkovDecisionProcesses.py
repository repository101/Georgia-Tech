import os
import pickle
import sys

import hiive.mdptoolbox.example
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from hiive import mdptoolbox
from hiive.mdptoolbox import mdp

from policy_iteration import run_policy_iteration
from q_learner import run_q_learner
from util import generate_frozen_lake, plot_combined_iteration_vs_reward_frozen, plot_iteration_vs_reward, \
	plot_iteration_vs_runtime, extract_data_to_frame, plot_combined_iteration_vs_runtime_frozen, check_folder, \
	plot_combined_iteration_vs_runtime_forest, plot_iteration_vs_error, plot_combined_iteration_vs_error_forest, \
	load_pickles, heatmap
from value_iteration import run_value_iteration

pd.options.mode.chained_assignment = "warn"

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500

plt.tight_layout()

save_dir = "Graphs"
mpl.rcParams['agg.path.chunksize'] = 10000

"""
https://learning.oreilly.com/library/view/reinforcement-learning-algorithms/9781789131116/ab06aa68-01f9-481e-94ac-4c6748c3b858.xhtml

env = gym.make('FrozenLake-v0')
env = env.unwrapped
num_actions = env.action_space.n
num_states = env.observation_space.n
V = np.zeros(num_states)
policy = np.zeros(num_states)
"""
Gen_Data = True


# noinspection DuplicatedCode
def run():
	try:
		np.random.seed(42)
		if Gen_Data:
			check_folder(save_dir)
			np.random.seed(42)
			env_small = generate_frozen_lake(size="s", is_slippery=False)
			env_small = env_small.unwrapped
			all_results = {"Frozen_Lake": {"Small": {}, "Large": {}}, "Forest_Management": {"Small": {}, "Large": {}}}
			max_iterations = 15000
			all_gammas = [0.6, 0.8, 0.9, 0.95, 0.99, 0.9999]
			all_epsilon = [0.00001, 0.0001, 0.001, 0.01, 0.1]

			all_dataframes = {}
			for sz in ["small", "large"]:
				for alg in ["policy_iteration", "value_iteration", "q_learner"]:
					for param in ["gamma", "eps"]:
						for prob in ["forest", "frozen"]:
							for result_type in ["reward", "runtime"]:
								name = f"{sz}_{alg}_{param}_{prob}_{result_type}"
								if param == "gamma":
									all_dataframes[name] = pd.DataFrame(index=np.arange(max_iterations),
																		columns=all_gammas,
																		data=np.zeros(
																			shape=(max_iterations, len(all_gammas))))
								else:
									all_dataframes[name] = pd.DataFrame(index=np.arange(max_iterations),
																		columns=all_epsilon,
																		data=np.zeros(
																			shape=(max_iterations, len(all_epsilon))))
			for eps in all_epsilon:
				print(f"Epsilon Currently: {eps}")
				gamma = 0.99
				problem = "Frozen Lake"
				prob_size = "Small"
				temp_eps = str(eps)[-1]
				str_eps = "0_" + (int(temp_eps) - 1) * "0" + "1"
				key = f"EPS_{str_eps}"
				all_results["Frozen_Lake"][f"{prob_size}"][key] = {"Value_Iteration": None,
																   "Policy_Iteration": None,
																   "Q_Learner": None}
				small_kwargs = {"env": env_small, "max_iterations": max_iterations, "gamma": gamma, "eps": eps}
				value_iteration_results_small = run_value_iteration(small_kwargs)
				value_iteration_results_small_df = extract_data_to_frame(run_stats={k: v for k, v in
																					value_iteration_results_small[
																						"Results"].items() if k
																					in ('Run_Times', 'Total_Value',
																						'Difference_Per_Iteration')},
																		 max_iterations=max_iterations,
																		 ending_iteration=
																		 value_iteration_results_small["Results"][
																			 "Number_Of_Iterations"])
				policy_iteration_results_small = run_policy_iteration(small_kwargs)
				policy_iteration_results_small_df = extract_data_to_frame(run_stats={k: v for k, v in
																					 policy_iteration_results_small[
																						 "Results"].items() if k
																					 in ('Run_Times', 'Total_Value')},
																		  max_iterations=max_iterations,
																		  ending_iteration=
																		  policy_iteration_results_small["Results"][
																			  "Number_Of_Iterations"] - 1)
				q_learner_results_small = run_q_learner(small_kwargs, test=True)
				q_learner_results_small_df = extract_data_to_frame(run_stats={k: v for k, v in
																			  q_learner_results_small["Results"].items()
																			  if k
																			  in ('Run_Times', 'Steps_Per_Iteration',
																				  'Rewards_Per_Iteration')},
																   max_iterations=max_iterations,
																   ending_iteration=q_learner_results_small["Results"][
																	   "Number_Of_Iterations"], is_q_learner=True)
				env_large = generate_frozen_lake(size="l", is_slippery=False)
				env_large = env_large.unwrapped
				large_kwargs = {"env": env_large, "max_iterations": max_iterations, "gamma": gamma}
				problem = "Frozen Lake"
				prob_size = "Large"
				all_results["Frozen_Lake"][f"{prob_size}"][key] = {"Value_Iteration": None,
																   "Policy_Iteration": None,
																   "Q_Learner": None}
				value_iteration_results_large = run_value_iteration(large_kwargs)
				value_iteration_results_large_df = extract_data_to_frame(run_stats={k: v for k, v in
																					value_iteration_results_large[
																						"Results"].items() if k
																					in ('Run_Times', 'Total_Value',
																						'Difference_Per_Iteration')},
																		 max_iterations=max_iterations,
																		 ending_iteration=
																		 value_iteration_results_large["Results"][
																			 "Number_Of_Iterations"])
				policy_iteration_results_large = run_policy_iteration(large_kwargs)
				policy_iteration_results_large_df = extract_data_to_frame(run_stats={k: v for k, v in
																					 policy_iteration_results_large[
																						 "Results"].items() if k
																					 in ('Run_Times', 'Total_Value')},
																		  max_iterations=max_iterations,
																		  ending_iteration=
																		  policy_iteration_results_large["Results"][
																			  "Number_Of_Iterations"] - 1)
				q_learner_results_large = run_q_learner(large_kwargs, is_large=True, test=True)
				q_learner_results_large_df = extract_data_to_frame(run_stats={k: v for k, v in
																			  q_learner_results_large["Results"].items()
																			  if k
																			  in ('Run_Times', 'Steps_Per_Iteration',
																				  'Rewards_Per_Iteration')},
																   max_iterations=max_iterations,
																   ending_iteration=q_learner_results_large["Results"][
																	   "Number_Of_Iterations"], is_q_learner=True)
				problem = "Forest Management"
				prob_size = "Small"
				all_results["Forest_Management"][f"{prob_size}"][key] = {"Value_Iteration": None,
																		 "Policy_Iteration": None,
																		 "Q_Learner": None}
				P_small, R_small = mdptoolbox.example.forest(S=10)
				vi_small = mdp.ValueIteration(P_small, R_small, max_iter=max_iterations, gamma=gamma)
				vi_small.run()
				vi_small_forest_management_df = extract_data_to_frame(vi_small.run_stats, max_iterations=max_iterations,
																	  ending_iteration=max_iterations, is_forest=True,
																	  increase_to_max=True)
				pi_small = mdp.PolicyIteration(P_small, R_small, max_iter=max_iterations, gamma=gamma)
				pi_small.run()
				pi_small_forest_management_df = extract_data_to_frame(pi_small.run_stats, max_iterations=max_iterations,
																	  ending_iteration=max_iterations, is_forest=True,
																	  increase_to_max=True)
				ql_small = mdp.QLearning(P_small, R_small, n_iter=max_iterations, gamma=gamma)
				ql_small.run()
				ql_small_forest_management_df = extract_data_to_frame(ql_small.run_stats, max_iterations=max_iterations,
																	  ending_iteration=max_iterations, is_forest=True,
																	  increase_to_max=True)
				problem = "Forest Management"
				prob_size = "Large"
				all_results["Forest_Management"][f"{prob_size}"][key] = {"Value_Iteration": None,
																		 "Policy_Iteration": None,
																		 "Q_Learner": None}
				P_large, R_large = mdptoolbox.example.forest(S=1000)
				vi_large = mdp.ValueIteration(P_large, R_large, max_iter=max_iterations, gamma=gamma)
				vi_large.run()
				vi_large_forest_management_df = extract_data_to_frame(vi_large.run_stats, max_iterations=max_iterations,
																	  ending_iteration=max_iterations, is_forest=True,
																	  increase_to_max=True)
				pi_large = mdp.PolicyIteration(P_large, R_large, max_iter=max_iterations, gamma=gamma)
				pi_large.run()
				pi_large_forest_management_df = extract_data_to_frame(pi_large.run_stats, max_iterations=max_iterations,
																	  ending_iteration=max_iterations, is_forest=True,
																	  increase_to_max=True)
				ql_large = mdp.QLearning(P_large, R_large, n_iter=max_iterations, gamma=gamma)
				ql_large.run()
				ql_large_forest_management_df = extract_data_to_frame(ql_large.run_stats, max_iterations=max_iterations,
																	  ending_iteration=max_iterations, is_forest=True,
																	  increase_to_max=True, is_q_learner=True)
				prob_size = "Small"
				all_dataframes[f"{prob_size.lower()}_value_iteration_eps_frozen_reward"].loc[:, eps] = \
					value_iteration_results_small_df["Total_Value"]
				all_dataframes[f"{prob_size.lower()}_value_iteration_eps_frozen_runtime"].loc[:, eps] = \
					value_iteration_results_small_df["Run_Times"]
				all_dataframes[f"{prob_size.lower()}_policy_iteration_eps_frozen_reward"].loc[:, eps] = \
					policy_iteration_results_small_df["Total_Value"]
				all_dataframes[f"{prob_size.lower()}_policy_iteration_eps_frozen_runtime"].loc[:, eps] = \
					policy_iteration_results_small_df["Run_Times"]
				all_dataframes[f"{prob_size.lower()}_q_learner_eps_frozen_reward"].loc[:, eps] = \
					q_learner_results_small_df["Rewards_Per_Iteration"]
				all_dataframes[f"{prob_size.lower()}_q_learner_eps_frozen_runtime"].loc[:, eps] = \
					q_learner_results_small_df["Run_Times"]
				all_dataframes[f"{prob_size.lower()}_value_iteration_eps_forest_reward"].loc[:, eps] = \
					vi_small_forest_management_df["Reward"]
				all_dataframes[f"{prob_size.lower()}_value_iteration_eps_forest_runtime"].loc[:, eps] = \
					vi_small_forest_management_df["Time"]
				all_dataframes[f"{prob_size.lower()}_policy_iteration_eps_forest_reward"].loc[:, eps] = \
					pi_small_forest_management_df["Reward"]
				all_dataframes[f"{prob_size.lower()}_policy_iteration_eps_forest_runtime"].loc[:, eps] = \
					pi_small_forest_management_df["Time"]
				all_dataframes[f"{prob_size.lower()}_q_learner_eps_forest_reward"].loc[:, eps] = \
					ql_small_forest_management_df["Reward"]
				all_dataframes[f"{prob_size.lower()}_q_learner_eps_forest_runtime"].loc[:, eps] = \
					ql_small_forest_management_df["Time"]
				prob_size = "Large"
				all_dataframes[f"{prob_size.lower()}_value_iteration_eps_frozen_reward"].loc[:, eps] = \
					value_iteration_results_large_df["Total_Value"]
				all_dataframes[f"{prob_size.lower()}_value_iteration_eps_frozen_runtime"].loc[:, eps] = \
					value_iteration_results_large_df["Run_Times"]
				all_dataframes[f"{prob_size.lower()}_policy_iteration_eps_frozen_reward"].loc[:, eps] = \
					policy_iteration_results_large_df["Total_Value"]
				all_dataframes[f"{prob_size.lower()}_policy_iteration_eps_frozen_runtime"].loc[:, eps] = \
					policy_iteration_results_large_df["Run_Times"]
				all_dataframes[f"{prob_size.lower()}_q_learner_eps_frozen_reward"].loc[:, eps] = \
					q_learner_results_large_df["Rewards_Per_Iteration"]
				all_dataframes[f"{prob_size.lower()}_q_learner_eps_frozen_runtime"].loc[:, eps] = \
					q_learner_results_large_df["Run_Times"]
				all_dataframes[f"{prob_size.lower()}_value_iteration_eps_forest_reward"].loc[:, eps] = \
					vi_large_forest_management_df["Reward"]
				all_dataframes[f"{prob_size.lower()}_value_iteration_eps_forest_runtime"].loc[:, eps] = \
					vi_large_forest_management_df["Time"]
				all_dataframes[f"{prob_size.lower()}_policy_iteration_eps_forest_reward"].loc[:, eps] = \
					pi_large_forest_management_df["Reward"]
				all_dataframes[f"{prob_size.lower()}_policy_iteration_eps_forest_runtime"].loc[:, eps] = \
					pi_large_forest_management_df["Time"]
				all_dataframes[f"{prob_size.lower()}_q_learner_eps_forest_reward"].loc[:, eps] = \
					ql_large_forest_management_df["Reward"]
				all_dataframes[f"{prob_size.lower()}_q_learner_eps_forest_runtime"].loc[:, eps] = \
					ql_large_forest_management_df["Time"]

				prob_size = "Small"
				all_results["Frozen_Lake"][f"{prob_size}"][key]["Value_Iteration"] = {
					"Dataframe": value_iteration_results_small_df,
					"Raw_Results": value_iteration_results_small}
				all_results["Frozen_Lake"][f"{prob_size}"][key]["Policy_Iteration"] = {
					"Dataframe": policy_iteration_results_small_df,
					"Raw_Results": policy_iteration_results_small}
				all_results["Frozen_Lake"][f"{prob_size}"][key]["Q_Learner"] = {
					"Dataframe": q_learner_results_small_df,
					"Raw_Results": q_learner_results_small}
				all_results["Forest_Management"][f"{prob_size}"][key]["Value_Iteration"] = {
					"Dataframe": vi_small_forest_management_df,
					"Raw_Results": vi_small.run_stats}
				all_results["Forest_Management"][f"{prob_size}"][key]["Policy_Iteration"] = {
					"Dataframe": pi_small_forest_management_df,
					"Raw_Results": pi_small.run_stats}
				all_results["Forest_Management"][f"{prob_size}"][key]["Q_Learner"] = {
					"Dataframe": ql_small_forest_management_df,
					"Raw_Results": ql_small.run_stats}
				prob_size = "Large"
				all_results["Frozen_Lake"][f"{prob_size}"][key]["Value_Iteration"] = {
					"Dataframe": value_iteration_results_large_df,
					"Raw_Results": value_iteration_results_large}
				all_results["Frozen_Lake"][f"{prob_size}"][key]["Policy_Iteration"] = {
					"Dataframe": policy_iteration_results_large_df,
					"Raw_Results": policy_iteration_results_large}
				all_results["Frozen_Lake"][f"{prob_size}"][key]["Q_Learner"] = {
					"Dataframe": q_learner_results_large_df,
					"Raw_Results": q_learner_results_large}
				all_results["Forest_Management"][f"{prob_size}"][key]["Value_Iteration"] = {
					"Dataframe": vi_large_forest_management_df,
					"Raw_Results": vi_large.run_stats}
				all_results["Forest_Management"][f"{prob_size}"][key]["Policy_Iteration"] = {
					"Dataframe": pi_large_forest_management_df,
					"Raw_Results": pi_large.run_stats}
				all_results["Forest_Management"][f"{prob_size}"][key]["Q_Learner"] = {
					"Dataframe": ql_large_forest_management_df,
					"Raw_Results": ql_large.run_stats}
				print()

			for gamma in all_gammas:
				print(f"Gamma Currently: {gamma}")
				problem = "Frozen Lake"
				prob_size = "Small"
				str_gamma = str(gamma).replace(".", "_")
				key = f"GAMMA_{str_gamma}"
				all_results["Frozen_Lake"][f"{prob_size}"][key] = {"Value_Iteration": None,
																   "Policy_Iteration": None,
																   "Q_Learner": None}
				small_kwargs = {"env": env_small, "max_iterations": max_iterations, "gamma": gamma}
				value_iteration_results_small = run_value_iteration(small_kwargs)
				value_iteration_results_small_df = extract_data_to_frame(run_stats={k: v for k, v in
																					value_iteration_results_small[
																						"Results"].items() if k
																					in ('Run_Times', 'Total_Value',
																						'Difference_Per_Iteration')},
																		 max_iterations=max_iterations,
																		 ending_iteration=
																		 value_iteration_results_small["Results"][
																			 "Number_Of_Iterations"])
				policy_iteration_results_small = run_policy_iteration(small_kwargs)
				policy_iteration_results_small_df = extract_data_to_frame(run_stats={k: v for k, v in
																					 policy_iteration_results_small[
																						 "Results"].items() if k
																					 in ('Run_Times', 'Total_Value')},
																		  max_iterations=max_iterations,
																		  ending_iteration=
																		  policy_iteration_results_small["Results"][
																			  "Number_Of_Iterations"] - 1)
				q_learner_results_small = run_q_learner(small_kwargs)
				q_learner_results_small_df = extract_data_to_frame(run_stats={k: v for k, v in
																			  q_learner_results_small["Results"].items()
																			  if k
																			  in ('Run_Times', 'Steps_Per_Iteration',
																				  'Rewards_Per_Iteration')},
																   max_iterations=max_iterations,
																   ending_iteration=q_learner_results_small["Results"][
																	   "Number_Of_Iterations"], is_q_learner=True)
				env_large = generate_frozen_lake(size="l", is_slippery=False)
				env_large = env_large.unwrapped
				large_kwargs = {"env": env_large, "max_iterations": max_iterations, "gamma": gamma}
				problem = "Frozen Lake"
				prob_size = "Large"
				all_results["Frozen_Lake"][f"{prob_size}"][key] = {"Value_Iteration": None,
																   "Policy_Iteration": None,
																   "Q_Learner": None}
				value_iteration_results_large = run_value_iteration(large_kwargs)
				value_iteration_results_large_df = extract_data_to_frame(run_stats={k: v for k, v in
																					value_iteration_results_large[
																						"Results"].items() if k
																					in ('Run_Times', 'Total_Value',
																						'Difference_Per_Iteration')},
																		 max_iterations=max_iterations,
																		 ending_iteration=
																		 value_iteration_results_large["Results"][
																			 "Number_Of_Iterations"])
				policy_iteration_results_large = run_policy_iteration(large_kwargs)
				policy_iteration_results_large_df = extract_data_to_frame(run_stats={k: v for k, v in
																					 policy_iteration_results_large[
																						 "Results"].items() if k
																					 in ('Run_Times', 'Total_Value')},
																		  max_iterations=max_iterations,
																		  ending_iteration=
																		  policy_iteration_results_large["Results"][
																			  "Number_Of_Iterations"] - 1)
				q_learner_results_large = run_q_learner(large_kwargs, is_large=True)
				q_learner_results_large_df = extract_data_to_frame(run_stats={k: v for k, v in
																			  q_learner_results_large["Results"].items()
																			  if k
																			  in ('Run_Times', 'Steps_Per_Iteration',
																				  'Rewards_Per_Iteration')},
																   max_iterations=max_iterations,
																   ending_iteration=q_learner_results_large["Results"][
																	   "Number_Of_Iterations"], is_q_learner=True)
				problem = "Forest Management"
				prob_size = "Small"
				all_results["Forest_Management"][f"{prob_size}"][key] = {"Value_Iteration": None,
																		 "Policy_Iteration": None,
																		 "Q_Learner": None}
				P_small, R_small = mdptoolbox.example.forest(S=10)
				vi_small = mdp.ValueIteration(P_small, R_small, max_iter=max_iterations, gamma=gamma)
				vi_small.run()
				vi_small_forest_management_df = extract_data_to_frame(vi_small.run_stats, max_iterations=max_iterations,
																	  ending_iteration=max_iterations, is_forest=True,
																	  increase_to_max=True)
				pi_small = mdp.PolicyIteration(P_small, R_small, max_iter=max_iterations, gamma=gamma)
				pi_small.run()
				pi_small_forest_management_df = extract_data_to_frame(pi_small.run_stats, max_iterations=max_iterations,
																	  ending_iteration=max_iterations, is_forest=True,
																	  increase_to_max=True)
				ql_small = mdp.QLearning(P_small, R_small, n_iter=max_iterations, gamma=gamma)
				ql_small.run()
				ql_small_forest_management_df = extract_data_to_frame(ql_small.run_stats, max_iterations=max_iterations,
																	  ending_iteration=max_iterations, is_forest=True,
																	  increase_to_max=True)
				problem = "Forest Management"
				prob_size = "Large"
				all_results["Forest_Management"][f"{prob_size}"][key] = {"Value_Iteration": None,
																		 "Policy_Iteration": None,
																		 "Q_Learner": None}
				P_large, R_large = mdptoolbox.example.forest(S=1000)
				vi_large = mdp.ValueIteration(P_large, R_large, max_iter=max_iterations, gamma=gamma)
				vi_large.run()
				vi_large_forest_management_df = extract_data_to_frame(vi_large.run_stats, max_iterations=max_iterations,
																	  ending_iteration=max_iterations, is_forest=True,
																	  increase_to_max=True)
				pi_large = mdp.PolicyIteration(P_large, R_large, max_iter=max_iterations, gamma=gamma)
				pi_large.run()
				pi_large_forest_management_df = extract_data_to_frame(pi_large.run_stats, max_iterations=max_iterations,
																	  ending_iteration=max_iterations, is_forest=True,
																	  increase_to_max=True)
				ql_large = mdp.QLearning(P_large, R_large, n_iter=max_iterations, gamma=gamma)
				ql_large.run()
				ql_large_forest_management_df = extract_data_to_frame(ql_large.run_stats, max_iterations=max_iterations,
																	  ending_iteration=max_iterations, is_forest=True,
																	  increase_to_max=True, is_q_learner=True)
				prob_size = "small"
				all_dataframes[f"{prob_size.lower()}_value_iteration_gamma_frozen_reward"].loc[:, gamma] = \
					value_iteration_results_small_df["Total_Value"]
				all_dataframes[f"{prob_size.lower()}_value_iteration_gamma_frozen_runtime"].loc[:, gamma] = \
					value_iteration_results_small_df["Run_Times"]
				all_dataframes[f"{prob_size.lower()}_policy_iteration_gamma_frozen_reward"].loc[:, gamma] = \
					policy_iteration_results_small_df["Total_Value"]
				all_dataframes[f"{prob_size.lower()}_policy_iteration_gamma_frozen_runtime"].loc[:, gamma] = \
					policy_iteration_results_small_df["Run_Times"]
				all_dataframes[f"{prob_size.lower()}_q_learner_gamma_frozen_reward"].loc[:, gamma] = \
					q_learner_results_small_df["Rewards_Per_Iteration"]
				all_dataframes[f"{prob_size.lower()}_q_learner_gamma_frozen_runtime"].loc[:, gamma] = \
					q_learner_results_small_df["Run_Times"]
				all_dataframes[f"{prob_size.lower()}_value_iteration_gamma_forest_reward"].loc[:, gamma] = \
					vi_small_forest_management_df["Reward"]
				all_dataframes[f"{prob_size.lower()}_value_iteration_gamma_forest_runtime"].loc[:, gamma] = \
					vi_small_forest_management_df["Time"]
				all_dataframes[f"{prob_size.lower()}_policy_iteration_gamma_forest_reward"].loc[:, gamma] = \
					pi_small_forest_management_df["Reward"]
				all_dataframes[f"{prob_size.lower()}_policy_iteration_gamma_forest_runtime"].loc[:, gamma] = \
					pi_small_forest_management_df["Time"]
				all_dataframes[f"{prob_size.lower()}_q_learner_gamma_forest_reward"].loc[:, gamma] = \
					ql_small_forest_management_df["Reward"]
				all_dataframes[f"{prob_size.lower()}_q_learner_gamma_forest_runtime"].loc[:, gamma] = \
					ql_small_forest_management_df["Time"]
				prob_size = "large"
				all_dataframes[f"{prob_size.lower()}_value_iteration_gamma_frozen_reward"].loc[:, gamma] = \
					value_iteration_results_large_df["Total_Value"]
				all_dataframes[f"{prob_size.lower()}_value_iteration_gamma_frozen_runtime"].loc[:, gamma] = \
					value_iteration_results_large_df["Run_Times"]
				all_dataframes[f"{prob_size.lower()}_policy_iteration_gamma_frozen_reward"].loc[:, gamma] = \
					policy_iteration_results_large_df["Total_Value"]
				all_dataframes[f"{prob_size.lower()}_policy_iteration_gamma_frozen_runtime"].loc[:, gamma] = \
					policy_iteration_results_large_df["Run_Times"]
				all_dataframes[f"{prob_size.lower()}_q_learner_gamma_frozen_reward"].loc[:, gamma] = \
					q_learner_results_large_df["Rewards_Per_Iteration"]
				all_dataframes[f"{prob_size.lower()}_q_learner_gamma_frozen_runtime"].loc[:, gamma] = \
					q_learner_results_large_df["Run_Times"]
				all_dataframes[f"{prob_size.lower()}_value_iteration_gamma_forest_reward"].loc[:, gamma] = \
					vi_large_forest_management_df["Reward"]
				all_dataframes[f"{prob_size.lower()}_value_iteration_gamma_forest_runtime"].loc[:, gamma] = \
					vi_large_forest_management_df["Time"]
				all_dataframes[f"{prob_size.lower()}_policy_iteration_gamma_forest_reward"].loc[:, gamma] = \
					pi_large_forest_management_df["Reward"]
				all_dataframes[f"{prob_size.lower()}_policy_iteration_gamma_forest_runtime"].loc[:, gamma] = \
					pi_large_forest_management_df["Time"]
				all_dataframes[f"{prob_size.lower()}_q_learner_gamma_forest_reward"].loc[:, gamma] = \
					ql_large_forest_management_df["Reward"]
				all_dataframes[f"{prob_size.lower()}_q_learner_gamma_forest_runtime"].loc[:, gamma] = \
					ql_large_forest_management_df["Time"]

				prob_size = "Small"
				all_results["Frozen_Lake"][f"{prob_size}"][key]["Value_Iteration"] = {
					"Dataframe": value_iteration_results_small_df,
					"Raw_Results": value_iteration_results_small}
				all_results["Frozen_Lake"][f"{prob_size}"][key]["Policy_Iteration"] = {
					"Dataframe": policy_iteration_results_small_df,
					"Raw_Results": policy_iteration_results_small}
				all_results["Frozen_Lake"][f"{prob_size}"][key]["Q_Learner"] = {
					"Dataframe": q_learner_results_small_df,
					"Raw_Results": q_learner_results_small}
				all_results["Forest_Management"][f"{prob_size}"][key]["Value_Iteration"] = {
					"Dataframe": vi_small_forest_management_df,
					"Raw_Results": vi_small.run_stats}
				all_results["Forest_Management"][f"{prob_size}"][key]["Policy_Iteration"] = {
					"Dataframe": pi_small_forest_management_df,
					"Raw_Results": pi_small.run_stats}
				all_results["Forest_Management"][f"{prob_size}"][key]["Q_Learner"] = {
					"Dataframe": ql_small_forest_management_df,
					"Raw_Results": ql_small.run_stats}
				prob_size = "Large"
				all_results["Frozen_Lake"][f"{prob_size}"][key]["Value_Iteration"] = {
					"Dataframe": value_iteration_results_large_df,
					"Raw_Results": value_iteration_results_large}
				all_results["Frozen_Lake"][f"{prob_size}"][key]["Policy_Iteration"] = {
					"Dataframe": policy_iteration_results_large_df,
					"Raw_Results": policy_iteration_results_large}
				all_results["Frozen_Lake"][f"{prob_size}"][key]["Q_Learner"] = {
					"Dataframe": q_learner_results_large_df,
					"Raw_Results": q_learner_results_large}
				all_results["Forest_Management"][f"{prob_size}"][key]["Value_Iteration"] = {
					"Dataframe": vi_large_forest_management_df,
					"Raw_Results": vi_large.run_stats}
				all_results["Forest_Management"][f"{prob_size}"][key]["Policy_Iteration"] = {
					"Dataframe": pi_large_forest_management_df,
					"Raw_Results": pi_large.run_stats}
				all_results["Forest_Management"][f"{prob_size}"][key]["Q_Learner"] = {
					"Dataframe": ql_large_forest_management_df,
					"Raw_Results": ql_large.run_stats}
				print()
			for sz in ["small", "large"]:
				for alg in ["policy_iteration", "value_iteration", "q_learner"]:
					for param in ["gamma", "eps"]:
						for prob in ["forest", "frozen"]:
							for result_type in ["reward", "runtime"]:
								name = f"{sz}_{alg}_{param}_{prob}_{result_type}"
								with open(f"{os.getcwd()}/{name}.pkl", "wb") as output_file:
									pickle.dump(all_dataframes[f"{name}"], output_file)
									output_file.close()

			with open(f"{os.getcwd()}/All_Results_From_All_Runs.pkl", "wb") as output_file:
				pickle.dump(all_results, output_file)
				output_file.close()
		exit()
		all_data = load_pickles()

		gamma = 0.99
		problem = "Frozen Lake"
		prob_size = "Small"
		max_iterations = 15000
		env_small = generate_frozen_lake(size="s", is_slippery=False)
		env_small = env_small.unwrapped
		small_kwargs = {"env": env_small, "max_iterations": max_iterations, "gamma": gamma}
		value_iteration_results_small = run_value_iteration(small_kwargs)
		value_iteration_results_small_df = extract_data_to_frame(run_stats={k: v for k, v in
																			value_iteration_results_small[
																				"Results"].items() if k
																			in ('Run_Times', 'Total_Value',
																				'Difference_Per_Iteration')},
																 max_iterations=max_iterations,
																 ending_iteration=
																 value_iteration_results_small["Results"][
																	 "Number_Of_Iterations"])
		policy_iteration_results_small = run_policy_iteration(small_kwargs)
		policy_iteration_results_small_df = extract_data_to_frame(run_stats={k: v for k, v in
																			 policy_iteration_results_small[
																				 "Results"].items() if k
																			 in ('Run_Times', 'Total_Value')},
																  max_iterations=max_iterations,
																  ending_iteration=
																  policy_iteration_results_small["Results"][
																	  "Number_Of_Iterations"] - 1)
		q_learner_results_small = run_q_learner(small_kwargs)
		q_learner_results_small_df = extract_data_to_frame(run_stats={k: v for k, v in
																	  q_learner_results_small["Results"].items()
																	  if k
																	  in ('Run_Times', 'Steps_Per_Iteration',
																		  'Rewards_Per_Iteration')},
														   max_iterations=max_iterations,
														   ending_iteration=q_learner_results_small["Results"][
															   "Number_Of_Iterations"], is_q_learner=True)
		env_large = generate_frozen_lake(size="l", is_slippery=False)
		env_large = env_large.unwrapped
		large_kwargs = {"env": env_large, "max_iterations": max_iterations, "gamma": gamma}
		problem = "Frozen Lake"
		prob_size = "Large"
		value_iteration_results_large = run_value_iteration(large_kwargs)
		value_iteration_results_large_df = extract_data_to_frame(run_stats={k: v for k, v in
																			value_iteration_results_large[
																				"Results"].items() if k
																			in ('Run_Times', 'Total_Value',
																				'Difference_Per_Iteration')},
																 max_iterations=max_iterations,
																 ending_iteration=
																 value_iteration_results_large["Results"][
																	 "Number_Of_Iterations"])
		policy_iteration_results_large = run_policy_iteration(large_kwargs)
		policy_iteration_results_large_df = extract_data_to_frame(run_stats={k: v for k, v in
																			 policy_iteration_results_large[
																				 "Results"].items() if k
																			 in ('Run_Times', 'Total_Value')},
																  max_iterations=max_iterations,
																  ending_iteration=
																  policy_iteration_results_large["Results"][
																	  "Number_Of_Iterations"] - 1)
		q_learner_results_large = run_q_learner(large_kwargs, is_large=True)
		q_learner_results_large_df = extract_data_to_frame(run_stats={k: v for k, v in
																	  q_learner_results_large["Results"].items()
																	  if k
																	  in ('Run_Times', 'Steps_Per_Iteration',
																		  'Rewards_Per_Iteration')},
														   max_iterations=max_iterations,
														   ending_iteration=q_learner_results_large["Results"][
															   "Number_Of_Iterations"], is_q_learner=True)
		problem = "Forest Management"
		prob_size = "Small"
		P_small, R_small = mdptoolbox.example.forest(S=10)
		vi_small = mdp.ValueIteration(P_small, R_small, max_iter=max_iterations, gamma=gamma)
		vi_small.run()
		vi_small_forest_management_df = extract_data_to_frame(vi_small.run_stats, max_iterations=max_iterations,
															  ending_iteration=max_iterations, is_forest=True,
															  increase_to_max=True)
		pi_small = mdp.PolicyIteration(P_small, R_small, max_iter=max_iterations, gamma=gamma)
		pi_small.run()
		pi_small_forest_management_df = extract_data_to_frame(pi_small.run_stats, max_iterations=max_iterations,
															  ending_iteration=max_iterations, is_forest=True,
															  increase_to_max=True)
		ql_small = mdp.QLearning(P_small, R_small, n_iter=max_iterations, gamma=gamma)
		ql_small.run()
		ql_small_forest_management_df = extract_data_to_frame(ql_small.run_stats, max_iterations=max_iterations,
															  ending_iteration=max_iterations, is_forest=True,
															  increase_to_max=True)
		problem = "Forest Management"
		prob_size = "Large"
		P_large, R_large = mdptoolbox.example.forest(S=1000)
		vi_large = mdp.ValueIteration(P_large, R_large, max_iter=max_iterations, gamma=gamma)
		vi_large.run()
		vi_large_forest_management_df = extract_data_to_frame(vi_large.run_stats, max_iterations=max_iterations,
															  ending_iteration=max_iterations, is_forest=True,
															  increase_to_max=True)
		pi_large = mdp.PolicyIteration(P_large, R_large, max_iter=max_iterations, gamma=gamma)
		pi_large.run()
		pi_large_forest_management_df = extract_data_to_frame(pi_large.run_stats, max_iterations=max_iterations,
															  ending_iteration=max_iterations, is_forest=True,
															  increase_to_max=True)
		ql_large = mdp.QLearning(P_large, R_large, n_iter=max_iterations, gamma=gamma)
		ql_large.run()
		ql_large_forest_management_df = extract_data_to_frame(ql_large.run_stats, max_iterations=max_iterations,
															  ending_iteration=max_iterations, is_forest=True,
															  increase_to_max=True, is_q_learner=True)
		problem = "Frozen Lake"
		prob_size = "Small"
		plot_iteration_vs_reward(dataframe=value_iteration_results_small_df,
								 title=f"{problem} - {prob_size} \nValue Iteration\nReward Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=True,
								 filename=f"{problem}_Value_Iteration_{prob_size}", legend_label="VI")
		plot_iteration_vs_reward(dataframe=value_iteration_results_small_df,
								 title=f"Value Iteration {problem} - {prob_size} \nValue Iteration\nReward and "
									   f"Runtime Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=True,
								 filename=f"{problem}_Value_Iteration_{prob_size}", legend_label="VI",
								 plot_runtime=True)
		plot_iteration_vs_runtime(dataframe=value_iteration_results_small_df,
								  title=f"{problem} - {prob_size} \nValue Iteration\nRuntime Vs Iteration",
								  xlabel="Iteration", ylabel="Runtime", is_frozen=True,
								  filename=f"{problem}_Value_Iteration_{prob_size}", legend_label="VI")
		plot_iteration_vs_reward(dataframe=policy_iteration_results_small_df,
								 title=f"{problem} - {prob_size} \nPolicy Iteration\nReward Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=True,
								 filename=f"{problem}_Policy_Iteration_{prob_size}", legend_label="PI")
		plot_iteration_vs_reward(dataframe=policy_iteration_results_small_df,
								 title=f"{problem} - {prob_size} \nPolicy Iteration\nReward and Runtime Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=True,
								 filename=f"{problem}_Policy_Iteration_{prob_size}", legend_label="PI",
								 plot_runtime=True)
		plot_iteration_vs_runtime(dataframe=policy_iteration_results_small_df,
								  title=f"{problem} - {prob_size} \nPolicy Iteration\nRuntime Vs Iteration",
								  xlabel="Iteration", ylabel="Runtime", is_frozen=True,
								  filename=f"{problem}_Policy_Iteration_{prob_size}", legend_label="PI")
		plot_iteration_vs_reward(dataframe=q_learner_results_small_df,
								 title=f"{problem} - {prob_size} \nQ Learner\nReward Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=False,
								 filename=f"{problem}_Q_Learner_{prob_size}", is_frozen_q=True,
								 legend_label="QL")
		plot_iteration_vs_reward(dataframe=q_learner_results_small_df,
								 title=f"{problem} - {prob_size} \nQ Learner\nReward and Runtime Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=False,
								 filename=f"{problem}_Q_Learner_{prob_size}", is_frozen_q=True,
								 legend_label="QL", plot_runtime=True)
		plot_iteration_vs_runtime(dataframe=q_learner_results_small_df,
								  title=f"{problem} - {prob_size} \nQ Learner\nRuntime Vs Iteration",
								  xlabel="Iteration", ylabel="Runtime", is_frozen=False,
								  filename=f"{problem}_Q_Learner_{prob_size}", is_frozen_q=True,
								  legend_label="QL")
		plot_combined_iteration_vs_reward_frozen(value_iteration_results_small_df, policy_iteration_results_small_df,
												 q_learner_results_small_df,
												 title=f"Frozen Lake\nReward Vs Iteration Combined - {prob_size}",
												 xlabel="Iteration", ylabel="Reward",
												 filename=f"{prob_size}_problem",
												 max_iterations=q_learner_results_small["Results"][
													 "Number_Of_Iterations"])
		plot_combined_iteration_vs_runtime_frozen(value_iteration_results_small_df, policy_iteration_results_small_df,
												  q_learner_results_small_df,
												  title=f"Frozen Lake\nRuntime Vs Iteration Combined - {prob_size}",
												  xlabel="Iteration", ylabel="Runtime",
												  filename=f"{prob_size}_problem",
												  max_iterations=q_learner_results_small["Results"][
													  "Number_Of_Iterations"])
		problem = "Frozen Lake"
		prob_size = "Large"
		plot_iteration_vs_reward(dataframe=value_iteration_results_large_df,
								 title=f"{problem} - {prob_size} \nValue Iteration\nReward Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=True,
								 filename=f"{problem}_Value_Iteration_{prob_size}", legend_label="VI")
		plot_iteration_vs_reward(dataframe=value_iteration_results_large_df,
								 title=f"Value Iteration {problem} - {prob_size} \nValue Iteration\nReward and "
									   f"Runtime Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=True,
								 filename=f"{problem}_Value_Iteration_{prob_size}", legend_label="VI",
								 plot_runtime=True)
		plot_iteration_vs_runtime(dataframe=value_iteration_results_large_df,
								  title=f"{problem} - {prob_size} \nValue Iteration\nRuntime Vs Iteration",
								  xlabel="Iteration", ylabel="Runtime", is_frozen=True,
								  filename=f"{problem}_Value_Iteration_{prob_size}", legend_label="VI")
		plot_iteration_vs_reward(dataframe=policy_iteration_results_large_df,
								 title=f"{problem} - {prob_size} \nPolicy Iteration\nReward Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=True,
								 filename=f"{problem}_Policy_Iteration_{prob_size}", legend_label="PI")
		plot_iteration_vs_reward(dataframe=policy_iteration_results_large_df,
								 title=f"{problem} - {prob_size} \nPolicy Iteration\nReward and Runtime Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=True,
								 filename=f"{problem}_Policy_Iteration_{prob_size}", legend_label="PI",
								 plot_runtime=True)
		plot_iteration_vs_runtime(dataframe=policy_iteration_results_large_df,
								  title=f"{problem} - {prob_size} \nPolicy Iteration\nRuntime Vs Iteration",
								  xlabel="Iteration", ylabel="Runtime", is_frozen=True,
								  filename=f"{problem}_Policy_Iteration_{prob_size}", legend_label="PI")
		plot_iteration_vs_reward(dataframe=q_learner_results_large_df,
								 title=f"{problem} - {prob_size} \nQ Learner\nReward Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=False,
								 filename=f"{problem}_Q_Learner_{prob_size}", is_frozen_q=True,
								 legend_label="QL")
		plot_iteration_vs_reward(dataframe=q_learner_results_large_df,
								 title=f"{problem} - {prob_size} \nQ Learner\nReward and Runtime Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=False,
								 filename=f"{problem}_Q_Learner_{prob_size}", is_frozen_q=True,
								 legend_label="QL", plot_runtime=True)
		plot_iteration_vs_runtime(dataframe=q_learner_results_large_df,
								  title=f"{problem} - {prob_size} \nQ Learner\nRuntime Vs Iteration",
								  xlabel="Iteration", ylabel="Runtime", is_frozen=False,
								  filename=f"{problem}_Q_Learner_{prob_size}", is_frozen_q=True,
								  legend_label="QL")
		plot_combined_iteration_vs_reward_frozen(value_iteration_results_large_df, policy_iteration_results_large_df,
												 q_learner_results_large_df,
												 title=f"Frozen Lake\nReward Vs Iteration Combined - {prob_size}",
												 xlabel="Iteration", ylabel="Reward",
												 filename=f"{prob_size}_problem",
												 max_iterations=q_learner_results_large["Results"][
													 "Number_Of_Iterations"])
		plot_combined_iteration_vs_runtime_frozen(value_iteration_results_large_df, policy_iteration_results_large_df,
												  q_learner_results_large_df,
												  title=f"Frozen Lake\nRuntime Vs Iteration Combined - {prob_size}",
												  xlabel="Iteration", ylabel="Runtime",
												  filename=f"{prob_size}_problem",
												  max_iterations=q_learner_results_large["Results"][
													  "Number_Of_Iterations"])
		problem = "Forest Management"
		prob_size = "Small"
		plot_iteration_vs_reward(dataframe=vi_small_forest_management_df,
								 title=f"{problem} - {prob_size}\nValue Iteration \nReward Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=False, legend_label="VI",
								 filename=f"{problem}_Value_Iteration_{prob_size}", is_forest=True)
		plot_iteration_vs_error(dataframe=vi_small_forest_management_df,
								title=f"{problem} - {prob_size}\nValue Iteration \nError Vs Iteration",
								xlabel="Iteration", ylabel="Error", legend_label="VI",
								filename=f"{problem}_Value_Iteration_{prob_size}")
		plot_iteration_vs_error(dataframe=vi_small_forest_management_df,
								title=f"{problem} - {prob_size}\nValue Iteration \nError and Runtime Vs Iteration",
								xlabel="Iteration", ylabel="Error", legend_label="VI",
								filename=f"{problem}_Value_Iteration_{prob_size}", plot_runtime=True)
		plot_iteration_vs_reward(dataframe=pi_small_forest_management_df,
								 title=f"{problem} - {prob_size}\nPolicy Iteration \nReward Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=False, legend_label="PI",
								 filename=f"{problem}_Policy_Iteration_{prob_size}", is_forest=True)
		plot_iteration_vs_error(dataframe=pi_small_forest_management_df,
								title=f"{problem} - {prob_size}\nPolicy Iteration \nError Vs Iteration",
								xlabel="Iteration", ylabel="Error", legend_label="PI",
								filename=f"{problem}_Policy_Iteration_{prob_size}")
		plot_iteration_vs_error(dataframe=pi_small_forest_management_df,
								title=f"{problem} - {prob_size}\nPolicy Iteration \nError and Runtime Vs Iteration",
								xlabel="Iteration", ylabel="Error", legend_label="PI",
								filename=f"{problem}_Policy_Iteration_{prob_size}", plot_runtime=True)
		plot_iteration_vs_reward(dataframe=ql_small_forest_management_df,
								 title=f"{problem} - {prob_size}\nQ Learner\nReward Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=False, legend_label="QL",
								 filename=f"{problem}_Q_Learner_{prob_size}", is_forest=True)
		plot_iteration_vs_error(dataframe=ql_small_forest_management_df,
								title=f"{problem} - {prob_size}\nQ Learner \nError Vs Iteration",
								xlabel="Iteration", ylabel="Error", legend_label="QL",
								filename=f"{problem}_Q_Learner_{prob_size}")
		plot_iteration_vs_error(dataframe=ql_small_forest_management_df,
								title=f"{problem} - {prob_size} \nQ Learner\nError and Runtime Vs Iteration",
								xlabel="Iteration", ylabel="Error", legend_label="QL",
								filename=f"{problem}_Q_Learner_{prob_size}", plot_runtime=True)
		plot_combined_iteration_vs_runtime_forest(vi_small_forest_management_df, pi_small_forest_management_df,
												  ql_small_forest_management_df,
												  title=f"Forest Management\nRuntime Vs Iteration Combined - {prob_size}",
												  xlabel="Iteration", ylabel="Runtime",
												  filename=f"{prob_size}_problem", max_iterations=max_iterations)
		plot_combined_iteration_vs_error_forest(vi_small_forest_management_df, pi_small_forest_management_df,
												ql_small_forest_management_df,
												title=f"Forest Management\nError Vs Iteration Combined - {prob_size}",
												xlabel="Iteration", ylabel="Error",
												filename=f"{prob_size}_problem", max_iterations=max_iterations)
		problem = "Forest Management"
		prob_size = "Large"
		max_iterations = 15000
		plot_iteration_vs_reward(dataframe=vi_large_forest_management_df,
								 title=f"{problem} - {prob_size}\nValue Iteration\nReward Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=False, legend_label="VI",
								 filename=f"{problem}_Value_Iteration_{prob_size}", is_forest=True)
		plot_iteration_vs_error(dataframe=vi_large_forest_management_df,
								title=f"{problem} - {prob_size}\nValue Iteration \nError Vs Iteration",
								xlabel="Iteration", ylabel="Error", legend_label="VI",
								filename=f"{problem}_Value_Iteration_{prob_size}")
		plot_iteration_vs_error(dataframe=vi_large_forest_management_df,
								title=f"{problem} - {prob_size}\nValue Iteration \nError and Runtime Vs Iteration",
								xlabel="Iteration", ylabel="Error", legend_label="VI",
								filename=f"{problem}_Value_Iteration_{prob_size}", plot_runtime=True)
		plot_iteration_vs_reward(dataframe=pi_large_forest_management_df,
								 title=f"\nPolicy Iteration{problem} - {prob_size}\nPolicy Iteration \nReward Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=False, legend_label="PI",
								 filename=f"{problem}_Policy_Iteration_{prob_size}", is_forest=True)
		plot_iteration_vs_error(dataframe=pi_large_forest_management_df,
								title=f"{problem} - {prob_size}\nPolicy Iteration \nError Vs Iteration",
								xlabel="Iteration", ylabel="Error", legend_label="PI",
								filename=f"{problem}_Policy_Iteration_{prob_size}")
		plot_iteration_vs_error(dataframe=pi_large_forest_management_df,
								title=f"{problem} - {prob_size}\nPolicy Iteration \nError and Runtime Vs Iteration",
								xlabel="Iteration", ylabel="Error", legend_label="PI",
								filename=f"{problem}_Policy_Iteration_{prob_size}", plot_runtime=True)
		plot_iteration_vs_reward(dataframe=ql_large_forest_management_df,
								 title=f"{problem} - {prob_size}\nQ Learner \nReward Vs Iteration",
								 xlabel="Iteration", ylabel="Reward", is_frozen=False, legend_label="QL",
								 filename=f"{problem}_Q_Learner_{prob_size}", is_forest=True)
		plot_iteration_vs_error(dataframe=ql_large_forest_management_df,
								title=f"{problem} - {prob_size}\nQ Learner \nError Vs Iteration",
								xlabel="Iteration", ylabel="Error", legend_label="QL",
								filename=f"{problem}_Q_Learner_{prob_size}")
		plot_iteration_vs_error(dataframe=ql_large_forest_management_df,
								title=f"{problem} - {prob_size}\nQ Learner \nError and Runtime Vs Iteration",
								xlabel="Iteration", ylabel="Error", legend_label="QL",
								filename=f"{problem}_Q_Learner_{prob_size}", plot_runtime=True)
		plot_combined_iteration_vs_runtime_forest(vi_large_forest_management_df, pi_large_forest_management_df,
												  ql_large_forest_management_df,
												  title=f"Forest Management\nRuntime Vs Iteration Combined - {prob_size}",
												  xlabel="Iteration", ylabel="Runtime",
												  filename=f"{prob_size}_problem", max_iterations=max_iterations)
		plot_combined_iteration_vs_error_forest(vi_large_forest_management_df, pi_large_forest_management_df,
												ql_large_forest_management_df,
												title=f"Forest Management\nError Vs Iteration Combined - {prob_size}",
												xlabel="Iteration", ylabel="Error",
												filename=f"{prob_size}_problem", max_iterations=max_iterations)
		return
	except Exception as run_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'run'", run_exception)


def run_large_forest():
	try:
		max_iterations = 15000
		gamma = 0.99
		problem = "Forest Management"
		prob_size = "Large"
		P_large, R_large = mdptoolbox.example.forest(S=1000)
		vi_large = mdp.ValueIteration(P_large, R_large, max_iter=max_iterations, gamma=gamma)
		vi_large.run()
		vi_large_forest_management_df = extract_data_to_frame(vi_large.run_stats, max_iterations=max_iterations,
															  ending_iteration=max_iterations, is_forest=True,
															  increase_to_max=True, normalize=False)

		pi_large = mdp.PolicyIteration(P_large, R_large, max_iter=max_iterations, gamma=gamma)
		pi_large.run()
		pi_large_forest_management_df = extract_data_to_frame(pi_large.run_stats, max_iterations=max_iterations,
															  ending_iteration=max_iterations, is_forest=True,
															  increase_to_max=True, normalize=False)

		ql_large = mdp.QLearning(P_large, R_large, n_iter=max_iterations, gamma=gamma)
		ql_large.run()
		ql_large_forest_management_df = extract_data_to_frame(ql_large.run_stats, max_iterations=max_iterations,
															  ending_iteration=max_iterations, is_forest=True,
															  increase_to_max=True, is_q_learner=True, normalize=False)
		return vi_large_forest_management_df, pi_large_forest_management_df, ql_large_forest_management_df
	except Exception as run_large_forest_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'run_large_forest'", run_large_forest_exception)


def run_large_frozen():
	try:
		np.random.seed(42)
		max_iterations = 50000
		gamma = 0.99
		env_large = generate_frozen_lake(size="l", is_slippery=False)
		env_large = env_large.unwrapped
		large_kwargs = {"env": env_large, "max_iterations": max_iterations, "gamma": gamma}
		problem = "Frozen Lake"
		prob_size = "Large"
		value_iteration_results_large = run_value_iteration(large_kwargs)
		value_iteration_results_large_df = extract_data_to_frame(run_stats={k: v for k, v in
																			value_iteration_results_large[
																				"Results"].items() if k
																			in ('Run_Times', 'Total_Value',
																				'Difference_Per_Iteration')},
																 max_iterations=max_iterations,
																 ending_iteration=
																 value_iteration_results_large["Results"][
																	 "Number_Of_Iterations"])
		policy_iteration_results_large = run_policy_iteration(large_kwargs)
		policy_iteration_results_large_df = extract_data_to_frame(run_stats={k: v for k, v in
																			 policy_iteration_results_large[
																				 "Results"].items() if k
																			 in ('Run_Times', 'Total_Value')},
																  max_iterations=max_iterations,
																  ending_iteration=
																  policy_iteration_results_large["Results"][
																	  "Number_Of_Iterations"] - 1)
		value_iteration_optimal_policy = np.reshape(value_iteration_results_large["Results"]["Optimal_Policy"],
													newshape=(10, 10)).astype(int)
		policy_iteration_optimal_policy = np.reshape(policy_iteration_results_large["Results"]["Policy"],
													 newshape=(10, 10)).astype(int)
		value_iteration_optimal_policy[9, 0] = 3
		policy_iteration_optimal_policy[9, 0] = 3

		fig, ax = plt.subplots(figsize=(10, 8))
		heatmap(value_iteration_optimal_policy, np.arange(0, 10), np.arange(0, 10), ax=ax, annotate=True,
				title="Value Iteration Optimal Policy\n Frozen Lake")
		plt.savefig("Value_Iteration_Optimal_Policy_Frozen.png")
		plt.close("all")

		fig, ax = plt.subplots(figsize=(10, 8))
		heatmap(policy_iteration_optimal_policy, np.arange(0, 10), np.arange(0, 10), ax=ax, annotate=True,
				title="Policy Iteration Optimal Policy\n Frozen Lake")
		plt.savefig("Policy_Iteration_Optimal_Policy_Frozen.png")
		plt.close("all")

		q_learner_results_large = run_q_learner(large_kwargs, is_large=True)
		q_learner_results_large_df = extract_data_to_frame(run_stats={k: v for k, v in
																	  q_learner_results_large["Results"].items()
																	  if k
																	  in ('Run_Times', 'Steps_Per_Iteration',
																		  'Rewards_Per_Iteration')},
														   max_iterations=max_iterations,
														   ending_iteration=q_learner_results_large["Results"][
															   "Number_Of_Iterations"], is_q_learner=True)
		q_learner_optimal_policy = np.reshape(q_learner_results_large["Results"]["Optimal_Policy"],
											  newshape=(10, 10)).astype(int)
		q_learner_optimal_policy[9, 0] = 3
		fig, ax = plt.subplots(figsize=(10, 8))
		heatmap(q_learner_optimal_policy, np.arange(0, 10), np.arange(0, 10), ax=ax, annotate=True,
				title="Q-Learner Optimal Policy\n Frozen Lake")
		plt.savefig("Q_Learner_Optimal_Policy_Frozen.png")
		plt.close("all")

		plt.close("all")
		plt.style.use("ggplot")
		fig = plt.figure(figsize=(16, 6), constrained_layout=True)
		gs = gridspec.GridSpec(1, 3, width_ratios=[4, 4, 4], figure=fig, wspace=0.05, hspace=0.15)
		ax0 = plt.subplot(gs[0, 0])
		ax1 = plt.subplot(gs[0, 1])
		ax2 = plt.subplot(gs[0, 2])
		heatmap(value_iteration_optimal_policy, np.arange(0, 10), np.arange(0, 10), ax=ax0, annotate=True,
				title="Value Iteration Optimal Policy\n Frozen Lake", no_colorbar=True)
		heatmap(policy_iteration_optimal_policy, np.arange(0, 10), np.arange(0, 10), ax=ax1, annotate=True,
				title="Policy Iteration Optimal Policy\n Frozen Lake", no_colorbar=True)
		heatmap(q_learner_optimal_policy, np.arange(0, 10), np.arange(0, 10), ax=ax2, annotate=True,
				title="Q Learner Optimal Policy\n Frozen Lake", no_colorbar=True)
		plt.savefig("Combined_Optimal_Policies.png")

		return value_iteration_results_large, value_iteration_results_large_df, policy_iteration_results_large, \
			   policy_iteration_results_large_df, q_learner_results_large, q_learner_results_large_df
	except Exception as run_large_frozen_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'run_large_frozen'", run_large_frozen_exception)


if __name__ == "__main__":
	# run_large_forest()
	# raw_value_iteration, value_iteration_frozen_large, raw_policy_iteration, \
	# policy_iteration_frozen_large, raw_q_learner, q_learner_frozen_large = run_large_frozen()
	run()
	print()
