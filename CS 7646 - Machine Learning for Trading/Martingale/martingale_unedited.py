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


def test_code():
	win_prob = 0.60  # set appropriately to the probability of a win
	np.random.seed(gtid())  # do this only once
	print get_spin_result(win_prob)  # test the roulette spin


# add your code here to implement the experiments

def roulette_simulation(iterations=None, bankroll=None, episode_win=None):
	# american_casino, n = 2 because it uses 0(zero) and 00(double-zero)
	american_casino = True
	if american_casino:
		n_green = 2.0
	else:
		n_green = 1.0
	num_black = 18.0
	# win_prob = 0.60  # set appropriately to the probability of a win
	win_prob = (num_black / (36.0 + n_green))		# Formula from https://en.wikipedia.org/wiki/Roulette
	num_of_bets = 1000		# The max number of bets determined by Project 1 - Martingale
	episode_winnings = 0		# Winnings of an episode, where 1 episodes is 1 spin
	total_winnings = np.zeros([1, num_of_bets])		# Used to store a total after each episode
	episode_winnings_array = np.zeros([1, num_of_bets])		# Used to store the winnings for each episode
	count = 0		# Used to limit iterations, as a secondary failsafe
	Experiment_total_winnings = pd.DataFrame()
	Experiment_episode_winnings = pd.DataFrame()
	Experiment_bets = []
	return


if __name__ == "__main__":
	test_code()
