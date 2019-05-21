"""Assess a betting strategy. 			  		 			 	 	 		 		 	  		   	  			  	

Copyright 2018, Georgia Institute of Technology (Georgia Tech) 			  		 			 	 	 		 		 	  		   	  			  	
Atlanta, Georgia 30332 			  		 			 	 	 		 		 	  		   	  			  	
All Rights Reserved 			  		 			 	 	 		 		 	  		   	  			  	

Template code for CS 4646/7646 			  		 			 	 	 		 		 	  		   	  			  	

Georgia Tech asserts copyright ownership of this template and all derivative 			  		 			 	 	 		 		 	  		   	  			  	
works, including solutions to the projects assigned in this course. Students 			  		 			 	 	 		 		 	  		   	  			  	
and other users of this template code are advised not to share it with others 			  		 			 	 	 		 		 	  		   	  			  	
or to make it available on publicly viewable websites including repositories 			  		 			 	 	 		 		 	  		   	  			  	
such as github and gitlab.  This copyright statement should not be removed 			  		 			 	 	 		 		 	  		   	  			  	
or edited. 			  		 			 	 	 		 		 	  		   	  			  	

We do grant permission to share solutions privately with non-students such 			  		 			 	 	 		 		 	  		   	  			  	
as potential employers. However, sharing with other current or future 			  		 			 	 	 		 		 	  		   	  			  	
students of CS 7646 is prohibited and subject to being investigated as a 			  		 			 	 	 		 		 	  		   	  			  	
GT honor code violation. 			  		 			 	 	 		 		 	  		   	  			  	

-----do not edit anything above this line--- 			  		 			 	 	 		 		 	  		   	  			  	

Student Name: Josh Adams (replace with your name)
GT User ID: jadams334 (replace with your User ID)
GT ID: 903475599 (replace with your GT ID)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def author():
	return 'jadams334'  # replace tb34 with your Georgia Tech username.


def gtid():
	return 903475599  # replace with your GT ID number


def get_spin_result(win_prob):
	result = False
	if np.random.random() <= win_prob:
		result = True
	return result


def get_figure_1(total_df, episode_df, bets, win_prob):
	plt.figure(1)
	for i in range(10):
		# Experiment 1
		# Figure 1: 10 iterations, plot on chart X[0:300], y[-256:100]
		total, episode, count = roulette_simulation(episode_win=80, win_chance=win_prob)
		total_df = pd.concat([total_df, pd.DataFrame(total)], ignore_index=True)
		episode_df = pd.concat([episode_df, pd.DataFrame(episode)], ignore_index=True)
		bets.append(count)
	axes = plt.gca()
	axes.set_xlim([0, 300])
	axes.set_ylim([-256, 100])
	for n in range(10):
		plt.plot(episode_df.iloc[n, 0:300], label="Episode_{}".format(n))
	plt.title("Figure 1")
	plt.legend(["Episode_{}".format(t) for t in range(10)])
	plt.xlabel("Episodes")
	plt.ylabel("Winnings")
	plt.savefig('Figure_1.png')
	return total_df, episode_df, bets


def get_figure_2(win_prob, iterations=1000):
	plt.figure(2)
	total_df = pd.DataFrame(columns=[range(0, 1000)])
	episode_df = pd.DataFrame(columns=[range(0, 1000)])
	bets = []
	for i in range(iterations):
		# Experiment 1
		# Figure 2: 1000 iterations, plot on chart X[0:300], y[-256:100]
		total, episode, count = roulette_simulation(episode_win=80, win_chance=win_prob)
		total_df = pd.concat([total_df, pd.DataFrame(total)], ignore_index=True)
		episode_df = pd.concat([episode_df, pd.DataFrame(episode)], ignore_index=True)
		bets.append(count)
	mean_line = np.mean(episode_df.values, axis=0)     # This is the mean based on the rows
	std_per_spin = np.std(episode_df.values, axis=0)       # Standard Deviation based on the rows
	line_std_below_mean = mean_line - std_per_spin
	line_std_above_mean = mean_line + std_per_spin
	axes = plt.gca()
	axes.set_xlim([0, 300])
	axes.set_ylim([-256, 100])
	plt.plot(mean_line)
	plt.plot(line_std_above_mean)
	plt.plot(line_std_below_mean)
	plt.legend(("Mean", "STD_Above", "STD_Below"))
	plt.title("Figure 2")
	plt.xlabel("Episodes")
	plt.ylabel("Winnings")
	plt.savefig('Figure_2.png')
	get_figure_3(total_df=total_df, episode_df=episode_df, bets=bets)
	return


def get_figure_3(total_df, episode_df, bets):
	plt.figure(3)
	median_line = np.median(episode_df.values, axis=0)     # This is the mean based on the rows
	std_per_spin = np.std(episode_df.values, axis=0)       # Standard Deviation based on the rows
	line_std_below_mean = median_line - std_per_spin
	line_std_above_mean = median_line + std_per_spin
	axes = plt.gca()
	axes.set_xlim([0, 300])
	axes.set_ylim([-256, 100])
	plt.plot(median_line)
	plt.plot(line_std_above_mean)
	plt.plot(line_std_below_mean)
	plt.legend(("Median", "STD_Above", "STD_Below"))
	plt.title("Figure 3")
	plt.xlabel("Episodes")
	plt.ylabel("Winnings")
	plt.savefig('Figure_3.png')
	return


def get_figure_4(iterations, win_prob, bankroll=-256):
	plt.figure(4)
	total_df = pd.DataFrame(columns=[range(0, 1000)])
	episode_df = pd.DataFrame(columns=[range(0, 1000)])
	bets = []
	for i in range(iterations):
		# Experiment 2
		# Figure 4: 1000 iterations, plot on chart X[0:300], y[-256:100]
		total, episode, count = roulette_simulation(bankroll=bankroll, win_chance=win_prob)
		total_df = pd.concat([total_df, pd.DataFrame(total)], ignore_index=True)
		episode_df = pd.concat([episode_df, pd.DataFrame(episode)], ignore_index=True)
		bets.append(count)
	mean_line = np.mean(episode_df.values, axis=0)     # This is the mean based on the rows
	std_per_spin = np.std(episode_df.values, axis=0)       # Standard Deviation based on the rows
	line_std_below_mean = mean_line - std_per_spin
	line_std_above_mean = mean_line + std_per_spin
	axes = plt.gca()
	axes.set_xlim([0, 300])
	axes.set_ylim([-256, 100])
	plt.plot(mean_line)
	plt.plot(line_std_above_mean)
	plt.plot(line_std_below_mean)
	plt.legend(("Mean", "STD_Above", "STD_Below"))
	plt.title("Figure 4")
	plt.xlabel("Episodes")
	plt.ylabel("Winnings")
	plt.savefig('Figure_4.png')
	get_figure_5(totalDataframe=total_df, episodeDataframe=episode_df, bets=bets)
	return


def get_figure_5(totalDataframe, episodeDataframe, bets):
	plt.figure(5)
	median_line = np.median(episodeDataframe.values, axis=0)  # This is the mean based on the rows
	std_per_spin = np.std(episodeDataframe.values, axis=0)  # Standard Deviation based on the rows
	line_std_below_mean = median_line - std_per_spin
	line_std_above_mean = median_line + std_per_spin
	axes = plt.gca()
	axes.set_xlim([0, 300])
	axes.set_ylim([-256, 100])
	plt.plot(median_line)
	plt.plot(line_std_above_mean)
	plt.plot(line_std_below_mean)
	plt.legend(("Median", "STD_Above", "STD_Below"))
	plt.title("Figure 5")
	plt.xlabel("Episodes")
	plt.ylabel("Winnings")
	plt.savefig('Figure_5.png')
	return


def test_code():
	Experiment_total_winnings = pd.DataFrame(columns=[range(0, 1000)])  # Dataframe to hold the total winnings from each iteration
	Experiment_episode_winnings = pd.DataFrame(columns=[range(0, 1000)])  # Dataframe to hold the winnings per episode
	Experiment_bets = []  # A list to keep track of the number of bets that occurred per iteration
	# american_casino, n = 2 because it uses 0(zero) and 00(double-zero)
	american_casino = True
	if american_casino:
		n_green = 2.0
	else:
		n_green = 1.0
	num_black = 18.0
	# win_prob = 0.60  # set appropriately to the probability of a win
	win_prob = (num_black / (36.0 + n_green))  # Formula from https://en.wikipedia.org/wiki/Roulette
	np.random.seed(gtid())  # do this only once
	print get_spin_result(win_prob)  # test the roulette spin
	Experiment_total_winnings, \
	Experiment_episode_winnings, \
	Experiment_bets = get_figure_1(
		total_df=Experiment_total_winnings,
		episode_df=Experiment_episode_winnings,
		bets=Experiment_bets,
		win_prob=win_prob)

	get_figure_2(win_prob=win_prob, iterations=1000)        # Get_Figure_3 is inside of this function

	get_figure_4(iterations=1000, win_prob=win_prob, bankroll=-256)     # Get_Figure_5 is inside of this function

	print "Finished"


# add your code here to implement the experiments

def roulette_simulation(bankroll=None, episode_win=80, win_chance=0.6):
	num_of_bets = 1000  # The max number of bets determined by Project 1 - Martingale
	episode_winnings = 0  # Winnings of an episode, where 1 episodes is 1 spin
	total_winnings = np.zeros([1, num_of_bets])  # Used to store a total after each episode
	episode_winnings_array = np.zeros([1, num_of_bets])  # Used to store the winnings for each episode
	count = 0  # Used to limit iterations, as a secondary failsafe
	if bankroll is not None:
		while (total_winnings[0, count] > bankroll) and (episode_winnings < episode_win):
			won = False
			bet_amount = 1
			if count >= 999:
				break
			if total_winnings[0, count] <= bankroll:
				break
			while not won:
				if count >= 999:
					break
				if total_winnings[0, count] <= bankroll:
					break
				# wager bet_amount on black
				won = get_spin_result(win_prob=win_chance)
				if won:
					count += 1
					episode_winnings += bet_amount
					episode_winnings_array[0, count] = episode_winnings
					total_winnings[0, count] = total_winnings[0, count - 1] + episode_winnings
				else:
					count += 1
					episode_winnings -= bet_amount
					episode_winnings_array[0, count] = episode_winnings
					total_winnings[0, count] = total_winnings[0, count - 1] + episode_winnings
					bet_amount *= 2
					bet_amount = min(bet_amount, (total_winnings[0, count]+256))
		if episode_winnings < episode_win:
			episode_winnings_array[0, count:] = bankroll
		else:
			episode_winnings_array[0, count:] = episode_win

		total_winnings[0, count:] = total_winnings[0, count - 1]
	else:
		while episode_winnings < episode_win:
			won = False
			bet_amount = 1
			while not won:
				# wager bet_amount on black
				won = get_spin_result(win_prob=win_chance)
				if won:
					count += 1
					episode_winnings += bet_amount
					episode_winnings_array[0, count] = episode_winnings
					total_winnings[0, count] = total_winnings[0, count - 1] + episode_winnings
				else:
					count += 1
					episode_winnings -= bet_amount
					bet_amount *= 2
					episode_winnings_array[0, count] = episode_winnings
					total_winnings[0, count] = total_winnings[0, count - 1] + episode_winnings
		episode_winnings_array[0, count:] = episode_win
		total_winnings[0, count:] = total_winnings[0, count - 1]
	return total_winnings, episode_winnings_array, count


if __name__ == "__main__":
	test_code()
