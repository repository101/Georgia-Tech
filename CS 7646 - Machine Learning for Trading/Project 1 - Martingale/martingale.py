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
import matplotlib as plt
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


def test_code(target_winnings):
    # american_casino, n = 2 because it uses 0 and 00
    american_casino = True
    if american_casino:
        n_green = 2
    else:
        n_green = 1
    num_black = 18
    num_red = 18
    # win_prob = 0.60  # set appropriately to the probability of a win
    win_prob = (num_black/(36 + n_green))
    np.random.seed(gtid())  # do this only once
    # print get_spin_result(win_prob)  # test the roulette spin
    # add your code here to implement the experiments
    num_of_bets = 1000
    episode_winnings = 0
    winnings = np.zeros([1,num_of_bets])
    count = 0
    while episode_winnings < target_winnings:
        won = False
        bet_amount = 1
        if bet_amount > winnings[0, count]:
            bet_amount = winnings[0, count]
        while not won:
            while count < 1000:
                # TODO: 
                won = get_spin_result(win_prob)
                count += 1
                if won:
                    episode_winnings += bet_amount
                    winnings[0, count-1] = episode_winnings
                else:
                    episode_winnings -= bet_amount
                    try:
                        winnings[0, count-1] = episode_winnings
                    except Exception as err:
                        print err
                    bet_amount *= 2
    winnings[0, count:] = target_winnings
    return winnings

if __name__ == "__main__":
    # Sim1 is with 10 iterations
    Experiment_1 = pd.DataFrame()
    tmp = [test_code(80) for i in range(10)]
    print "HEY"
    
    # Sim2 is with 1000 iterations
    Experiment_2 = pd.DataFrame()
    
    
    test_code()
