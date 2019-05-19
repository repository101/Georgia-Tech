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
import matplotlib.pyplot as plt
import pandas as pd


def author():
    return 'jadams334'  # replace tb34 with your Georgia Tech username.


def gtid():
    return 903475599  # replace with your GT ID number


def get_spin_result(win_prob):
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def normalize_data(df):
    # This will cause all graphs to start from the same stop if needed
    return df / df.ix[0, :]


def test_code(iterations, target_episode_winnings=None, bankroll=None):
    # american_casino, n = 2 because it uses 0 and 00
    american_casino = True
    if american_casino:
        n_green = 2.0
    else:
        n_green = 1.0
    num_black = 18.0
    num_red = 18.0
    # win_prob = 0.60  # set appropriately to the probability of a win
    win_prob = (num_black/(36.0 + n_green))
    # print get_spin_result(win_prob)  # test the roulette spin
    # add your code here to implement the experiments

    num_of_bets = 1000
    episode_winnings = 0
    total_winnings = np.zeros([1, num_of_bets])
    episode_winnings_array = np.zeros([1, num_of_bets])
    count = 0
    Experiment_total_winnings = pd.DataFrame()
    Experiment_episode_winnings = pd.DataFrame()
    Experiment_bets = []

    if (target_episode_winnings is None) or (bankroll is not None):
        # we are in Experiment 2 so we need to stop when we lose X amount of money
        for episode in range(iterations):
            current_bank_value = total_winnings[0, count]
            while (current_bank_value > bankroll) and (count < num_of_bets):
                won = False
                bet_amount = 1
                # if bet_amount > winnings[0, count]:
                #     bet_amount = winnings[0, count]
                while (not won) and (count <= num_of_bets-1):
                    if count >= num_of_bets-1:
                        break
                    # TODO: Wager bet_amount on black ??
                    won = get_spin_result(win_prob)
                    count += 1
                    if won:
                        episode_winnings += bet_amount
                        total_winnings[0, count] = total_winnings[0, count - 1] + episode_winnings
                        episode_winnings_array[0, count] = episode_winnings
                    else:
                        episode_winnings -= bet_amount
                        total_winnings[0, count] = total_winnings[0, count - 1] + episode_winnings
                        episode_winnings_array[0, count] = episode_winnings
                        bet_amount *= 2
            print ""
            total_winnings[0, count:] = total_winnings[0, count - 1]
            episode_winnings_array[0, count:] = target_episode_winnings
            Experiment_total_winnings = pd.concat([Experiment_total_winnings, pd.DataFrame(total_winnings)], axis=0)
            Experiment_episode_winnings = pd.concat([Experiment_episode_winnings, pd.DataFrame(episode_winnings_array)], axis=0)
            Experiment_bets.append(count)
        print ""
    else:
        for episode in range(iterations):
            while episode_winnings < target_episode_winnings:
                won = False
                bet_amount = 1
                # if bet_amount > winnings[0, count]:
                #     bet_amount = winnings[0, count]
                while not won:
                    # TODO: Wager bet_amount on black ??
                    won = get_spin_result(win_prob)
                    count += 1
                    if won:
                        episode_winnings += bet_amount
                        total_winnings[0, count] = total_winnings[0, count-1] + episode_winnings
                        episode_winnings_array[0, count] = episode_winnings
                        print ""
                    else:
                        episode_winnings -= bet_amount
                        total_winnings[0, count] = total_winnings[0, count - 1] + episode_winnings
                        episode_winnings_array[0, count] = episode_winnings
                        bet_amount *= 2
            total_winnings[0, count:] = total_winnings[0, count-1]
            episode_winnings_array[0, count:] = target_episode_winnings
            Experiment_total_winnings = pd.concat([Experiment_total_winnings, pd.DataFrame(total_winnings)], axis=0)
            Experiment_episode_winnings = pd.concat([Experiment_episode_winnings, pd.DataFrame(episode_winnings_array)], axis=0)
            Experiment_bets.append(count)
    return Experiment_total_winnings, Experiment_episode_winnings, Experiment_bets


if __name__ == "__main__":
    np.random.seed(gtid())  # do this only once
    Experiment_1_total_winnings = pd.DataFrame()
    Experiment_1_episode_winnings = pd.DataFrame()
    Experiment_1_number_of_bets = pd.DataFrame()
    for i in range(3):
        total_winnings, episode_winnings, number_of_bets = test_code(1, 80, -256)
        Experiment_1_total_winnings = pd.concat([Experiment_1_total_winnings, pd.DataFrame(total_winnings)], axis=0)
        Experiment_1_episode_winnings = pd.concat([Experiment_1_episode_winnings, pd.DataFrame(episode_winnings)], axis=0)
        Experiment_1_number_of_bets = pd.concat([Experiment_1_number_of_bets, pd.DataFrame(number_of_bets)], axis=0)
    print "HEY"
    Experiment_1_episode_winnings.plot()
    plt.show()

    print ""
    #
    # # Sim2 is with 1000 iterations
    # Experiment_2 = pd.DataFrame()
    #
    # test_code()

