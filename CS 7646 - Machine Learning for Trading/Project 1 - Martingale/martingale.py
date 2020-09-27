""""""
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
    """  		  	   		     		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		     		  		  		    	 		 		   		 		  
    """
    return "jadams334"  # replace tb34 with your Georgia Tech username.


def gtid():
    """  		  	   		     		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		     		  		  		    	 		 		   		 		  
    """
    return 903475599  # replace with your GT ID number


def get_spin_result(win_prob):
    """  		  	   		     		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		     		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		     		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		     		  		  		    	 		 		   		 		  
    """
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result

def figure_1(win_prob):
    title = "Experiment 1 - Figure 1"
    xlabel = "Spin"
    ylabel = "Winnings"

    df_fig_1 = experiment_1(win_prob, runs=10, num_bets=300)
    ax = df_fig_1.plot(fontsize=12, xlim=(0, 300),
                       ylim=(-256, 100), grid=True, use_index=False, alpha=0.75)
    ax.set_title(title, fontsize=15, weight='bold')
    ax.set_xlabel(xlabel, fontsize=15, weight='bold')
    ax.set_ylabel(ylabel, fontsize=15, weight='bold')
    ax.text(-4, 80, "{:.0f}".format(80.0), color="g", ha="right", va="center", fontsize=10, alpha=0.75)
    plt.axhline(y=80, color="g", linestyle=(0, (5, 5)), alpha=0.5)
    plt.tight_layout()
    plt.savefig("Experiment_1_Figure_1.png")
    plt.close("all")
    return


# noinspection DuplicatedCode
def figure_2(win_prob):
    df_fig_2 = experiment_1(win_prob, runs=1000, num_bets=1000)
    standard_deviation = np.abs(df_fig_2.std(axis=1))
    test = df_fig_2.iloc[-1]
    a, b = np.unique(test, return_counts=True)
    percent = b/1000
    expected_value_array = a * percent
    expected_value = np.sum(expected_value_array)
    mean = df_fig_2.mean(axis=1)
    pos_std_line = mean + standard_deviation
    neg_std_line = mean - standard_deviation
    median = df_fig_2.median(axis=1)
    data = {"Mean": mean,
            "Mean +Std": pos_std_line,
            "Mean \N{MINUS SIGN}Std": neg_std_line}
    result_df = pd.DataFrame(data=data, index=df_fig_2.index)
    title = "Experiment 1 - Figure 2"
    xlabel = "Spin"
    ylabel = "Winnings"
    ax = result_df.plot(title=title, fontsize=12, xlim=(0, 300),
                        ylim=(-256, 100), grid=True, use_index=False, alpha=0.75)
    ax.set_title(title, fontsize=15, weight='bold')
    ax.set_xlabel(xlabel, fontsize=15, weight='bold')
    ax.set_ylabel(ylabel, fontsize=15, weight='bold')
    ax.text(-4, 80, "{:.0f}".format(80.0), color="g", ha="right", va="center", fontsize=10, alpha=0.75)
    plt.axhline(y=80, color="g", linestyle=(0, (5, 5)), alpha=0.5)
    plt.tight_layout()
    plt.savefig("Experiment_1_Figure_2.png")
    plt.close("all")

    return median, standard_deviation, df_fig_2


# noinspection DuplicatedCode
def figure_3(median, std, df):
    pos_std_line = median + std
    neg_std_line = median - std
    data = {"Median": median,
            "Median +Std": pos_std_line,
            "Median \N{MINUS SIGN}Std": neg_std_line}
    result_df = pd.DataFrame(data=data, index=df.index)
    title = "Experiment 1 - Figure 3"
    xlabel = "Spin"
    ylabel = "Winnings"
    ax = result_df.plot(title=title, fontsize=12, xlim=(0, 300),
                        ylim=(-256, 100), grid=True, use_index=False, alpha=0.75)
    ax.set_title(title, fontsize=15, weight='bold')
    ax.set_xlabel(xlabel, fontsize=15, weight='bold')
    ax.set_ylabel(ylabel, fontsize=15, weight='bold')
    ax.text(-4, 80, "{:.0f}".format(80.0), color="g", ha="right", va="center", fontsize=10, alpha=0.75)
    plt.axhline(y=80, color="g", linestyle=(0, (5, 5)), alpha=0.5)
    plt.tight_layout()
    plt.savefig("Experiment_1_Figure_3.png")
    plt.close("all")
    return


# noinspection DuplicatedCode
def figure_4(win_prob):
    df_fig_4 = experiment_2(win_prob, runs=1000, num_bets=1000)
    test = df_fig_4.iloc[-1]
    a,b = np.unique(test, return_counts=True)
    percent = b/1000
    expected_value_array = a*percent
    expected_value = np.sum(expected_value_array)

    standard_deviation = df_fig_4.std(axis=1)
    mean = df_fig_4.mean(axis=1)
    pos_std_line = mean + standard_deviation
    neg_std_line = mean - standard_deviation

    median = df_fig_4.median(axis=1)
    data = {"Mean": mean,
            "Mean +Std": pos_std_line,
            "Mean \N{MINUS SIGN}Std": neg_std_line}
    result_df = pd.DataFrame(data=data, index=df_fig_4.index)
    title = "Experiment 2 - Figure 4"
    xlabel = "Spin"
    ylabel = "Winnings"

    ax = result_df.plot(title=title, fontsize=12, xlim=(0, 300),
                        ylim=(-256, 100), grid=True, use_index=False, alpha=0.75)
    ax.set_title(title, fontsize=15, weight='bold')
    ax.set_xlabel(xlabel, fontsize=15, weight='bold')
    ax.set_ylabel(ylabel, fontsize=15, weight='bold')
    plt.tight_layout()
    plt.savefig("Experiment_2_Figure_4.png")
    plt.close("all")
    return median, standard_deviation, df_fig_4


# noinspection DuplicatedCode
def figure_5(median, std, df):
    pos_std_line = median + std
    neg_std_line = median - std
    data = {"Median": median,
            "Median +Std": pos_std_line,
            "Median \N{MINUS SIGN}Std": neg_std_line}
    result_df = pd.DataFrame(data=data, index=df.index)
    title = "Experiment 2 - Figure 5"
    xlabel = "Spin"
    ylabel = "Winnings"
    ax = result_df.plot(title=title, fontsize=12, xlim=(0, 300),
                        ylim=(-256, 100), grid=True, use_index=False, alpha=0.75)
    ax.text(-4, 80, "{:.0f}".format(80.0), color="g", ha="right", va="center", fontsize=10, alpha=0.75)
    plt.axhline(y=80, color="g", linestyle=(0, (5, 5)), alpha=0.5)
    ax.set_title(title, fontsize=15, weight='bold')
    ax.set_xlabel(xlabel, fontsize=15, weight='bold')
    ax.set_ylabel(ylabel, fontsize=15, weight='bold')
    plt.tight_layout()
    plt.savefig("Experiment_2_Figure_5.png")
    plt.close("all")
    return


def experiment_1(win_prob=(1/38)*18, runs=10, num_bets=1000):
    results = np.zeros(shape=(num_bets, runs), dtype=np.int32)
    for i in range(runs):
        episode_winnings = 0
        winnings = np.zeros(shape=(num_bets,), dtype=np.int32)
        total_wins = 0
        total_losses = 0
        count = 0
        while episode_winnings < 80:
            won = False
            bet_amount = 1
            winnings[count] = episode_winnings

            while not won:
                won = get_spin_result(win_prob)
                count += 1
                if won:
                    episode_winnings += bet_amount
                    winnings[count] = episode_winnings
                    total_wins += 1
                else:
                    episode_winnings -= bet_amount
                    bet_amount *= 2
                    winnings[count] = episode_winnings
                    total_losses += 1
        if count < num_bets:
            winnings[count+1::] = 80
        results[:, i] = winnings

    # create dataframe
    winnings_dataframe = pd.DataFrame(results,
                                      columns=["Simulation {}".format(i) for i in range(runs)],
                                      index=["Spin {}".format(i) for i in range(num_bets)])
    return winnings_dataframe


# noinspection DuplicatedCode
def experiment_2(win_prob=(1/38)*18, runs=10, num_bets=1000, bank_limit=256):
    results = np.zeros(shape=(num_bets, runs), dtype=np.int32)
    for i in range(runs):
        episode_winnings = 0
        winnings = np.zeros(shape=(num_bets,), dtype=np.int32)
        total_wins = 0
        total_losses = 0
        count = 0
        bankroll = bank_limit
        while episode_winnings < 80:
            if (count >= num_bets - 1) or (bankroll <= 0):
                break
            won = False
            bet_amount = 1
            winnings[count] = episode_winnings
            while not won:
                if count >= num_bets-1:
                    break
                if bankroll == 0:
                    break
                if bankroll - bet_amount <= 0:
                    bet_amount = bankroll
                bankroll -= bet_amount
                won = get_spin_result(win_prob)
                count += 1
                if won:
                    episode_winnings += bet_amount
                    bankroll += (bet_amount * 2)
                    winnings[count] = episode_winnings
                    total_wins += 1
                else:
                    episode_winnings -= bet_amount
                    bet_amount *= 2
                    winnings[count] = episode_winnings
                    total_losses += 1
        if count < num_bets-1:
            winnings[count+1::] = episode_winnings
        results[:, i] = winnings

    # create dataframe
    winnings_dataframe = pd.DataFrame(results,
                                      columns=["Simulation {}".format(i) for i in range(runs)],
                                      index=["Spin {}".format(i) for i in range(num_bets)])

    return winnings_dataframe

def misc():
    print("This is a random message which will be printed, one from a print statement and one from a function")
    return

def test_code():
    """  		  	   		     		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		     		  		  		    	 		 		   		 		  
    """
    number_of_black_values = 18
    total_number_of_values = 38

    win_prob = (1.0 / total_number_of_values) * number_of_black_values  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once
    # add your code here to implement the experiments

    plt.style.use('ggplot')

    #region Figure 1
    figure_1(win_prob)
    #endregion

    # region Figure 2
    median, fig_2_standard_deviation, figure_2_df = figure_2(win_prob)
    # endregion

    # region Figure 3
    figure_3(median, fig_2_standard_deviation, figure_2_df)
    # endregion

    # region Figure 4
    median, fig_4_standard_deviation, figure_4_df = figure_4(win_prob)
    # endregion

    # region Figure 5
    figure_5(median, fig_4_standard_deviation, figure_4_df)
    # endregion


if __name__ == "__main__":
    test_code()
