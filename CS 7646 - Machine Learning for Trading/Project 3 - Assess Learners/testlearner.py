""""""
"""  		  	   		     		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
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
"""

import math
import sys

import numpy as np

import LinRegLearner as lrl


import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import BagLearner as bg
import DTLearner as dt
import RTLearner as rt
import util


def data_to_dataframe(data, df_name=None, is_bag=False):
    if df_name is None:
        temp_corr_data = {"Training Set": data["Correlation"]["In Sample"],
                          "Testing Set": data["Correlation"]["Out Sample"]}
        corr_df = pd.DataFrame(data=temp_corr_data)

        temp_rmse_data = {"Training Set": data["RMSE"]["In Sample"],
                          "Testing Set": data["RMSE"]["Out Sample"]}
        rmse_df = pd.DataFrame(data=temp_rmse_data)

        if is_bag:
            temp_time_data = {"Training Times": data["Training Times"].flatten(),
                              "Query Times": data["Query Times"].flatten()}
        else:
            temp_time_data = {"Training Times": data["Training Times"].flatten(),
                              "Query Times": data["Query Times"].flatten(),
                              "Tree Depth": data["Tree Depth"].flatten()}
        time_df = pd.DataFrame(data=temp_time_data)

    else:
        temp_corr_data = {"Training Set": data[df_name]["Correlation"]["In Sample"],
                          "Testing Set": data[df_name]["Correlation"]["Out Sample"]}
        corr_df = pd.DataFrame(data=temp_corr_data)

        temp_rmse_data = {"Training Set": data[df_name]["RMSE"]["In Sample"],
                          "Testing Set": data[df_name]["RMSE"]["Out Sample"]}
        rmse_df = pd.DataFrame(data=temp_rmse_data)

        if is_bag:
            temp_time_data = {"Training Times": data[df_name]["Training Times"].flatten(),
                              "Query Times": data[df_name]["Query Times"].flatten()}
        else:
            temp_time_data = {"Training Times": data[df_name]["Training Times"].flatten(),
                              "Query Times": data[df_name]["Query Times"].flatten(),
                              "Tree Depth": data[df_name]["Tree Depth"].flatten()}

        time_df = pd.DataFrame(data=temp_time_data)

    return corr_df, rmse_df, time_df


def get_experiment_one_charts(data):
    learner_name = "Decision Tree"

    plt.close("all")
    plt.style.use('ggplot')

    fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    rmse_title = f"{learner_name} \nRMSE vs.Leaf Size"
    rmse_ax = data["RMSE"]["In Sample"].mean(axis=1).plot(ax=axes1[0], label="In Sample", grid=True, alpha=0.75)
    data["RMSE"]["Out Sample"].mean(axis=1).plot(ax=rmse_ax, label="Out Sample", grid=True, alpha=0.75)
    rmse_ax.set_xlabel("Leaf Size")
    rmse_ax.set_ylabel("RMSE")
    rmse_ax.set_title(rmse_title, weight='bold')
    rmse_ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
    rmse_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")

    corr_title = f"{learner_name} \nCorrelation vs. Leaf Size"
    corr_ax = data["Correlation"]["In Sample"].mean(axis=1).plot(ax=axes1[1], title=corr_title,
                                                                 label="In Sample", grid=True, alpha=0.75)
    data["Correlation"]["Out Sample"].mean(axis=1).plot(ax=corr_ax, label="Out Sample", grid=True, alpha=0.75)
    corr_ax.set_xlabel("Leaf Size")
    corr_ax.set_title(corr_title, weight='bold')
    corr_ax.set_ylabel("Correlation")
    corr_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")

    plt.tight_layout()
    plt.savefig(f"{fig_dir}Experiment One RMSE and Correlation Results.png")
    return


def get_experiment_two_charts(bag_data, save_individual=True):
    learner_name = "Bag Learner"
    # region Grouped Charts

    plt.close("all")
    plt.style.use('ggplot')
    fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    rmse_title = f"{learner_name} RMSE vs. Leaf Size"
    rmse_ax = bag_data["RMSE"]["In Sample"].mean(axis=1).plot(ax=axes1[0], label="In Sample", grid=True, alpha=0.75)
    bag_data["RMSE"]["Out Sample"].mean(axis=1).plot(ax=rmse_ax, label="Out Sample", grid=True, alpha=0.75)
    rmse_ax.set_xlabel("Leaf Size")
    rmse_ax.set_ylabel("RMSE")
    rmse_ax.set_title(rmse_title, weight='bold')
    rmse_ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
    rmse_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")

    corr_title = f"{learner_name} Correlation vs. Leaf Size"
    corr_ax = bag_data["Correlation"]["In Sample"].mean(axis=1).plot(ax=axes1[1], label="In Sample", grid=True,
                                                                     alpha=0.75)
    bag_data["Correlation"]["Out Sample"].mean(axis=1).plot(ax=corr_ax, label="Out Sample", grid=True, alpha=0.75)
    corr_ax.set_xlabel("Leaf Size")
    corr_ax.set_title(corr_title, weight='bold')
    corr_ax.set_ylabel("Correlation")
    corr_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}Experiment Two RMSE and Correlation Results.png")

    plt.close("all")
    time_title = f"{learner_name} Training Time vs. Leaf Size"
    time_ax = bag_data["Training Times"].mean(axis=1).plot(grid=True, alpha=0.75)
    time_ax.set_xlabel("Leaf Size")
    time_ax.set_title(time_title, weight='bold')
    time_ax.set_ylabel("Time (s)")
    time_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}Experiment Two Time Results.png")

    plt.close("all")
    query_title = f"{learner_name} Query Time vs. Leaf Size"
    query_ax = bag_data["Query Times"].plot(grid=True, alpha=0.75)
    query_ax.set_xlabel("Leaf Size")
    query_ax.set_title(query_title, weight='bold')
    query_ax.set_ylabel("Time (s)")
    query_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}Experiment Two Time Results.png")

    plt.close("all")
    acc_title = f"{learner_name} Accuracy vs. Leaf Size"
    acc_ax = bag_data["Metrics"]["In Sample Accuracy"].mean(axis=1).plot(label="In Sample", grid=True, alpha=0.75)
    bag_data["Metrics"]["Out Sample Accuracy"].mean(axis=1).plot(ax=acc_ax, label="Out Sample", grid=True, alpha=0.75)
    acc_ax.set_xlabel("Leaf Size")
    acc_ax.set_title(acc_title, weight='bold')
    acc_ax.set_ylabel("Accuracy")
    acc_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}Experiment Two Accuracy Results.png")

    # end region

    if save_individual:
        # region Individual Charts
        plt.close("all")
        plt.style.use('ggplot')
        rmse_title = f"{learner_name} RMSE vs. Leaf Size"
        rmse_ax = bag_data["RMSE"]["In Sample"].mean(axis=1).plot(label="In Sample", grid=True, alpha=0.75)
        bag_data["RMSE"]["Out Sample"].mean(axis=1).plot(ax=rmse_ax, label="Out Sample", grid=True, alpha=0.75)
        rmse_ax.set_xlabel("Leaf Size")
        rmse_ax.set_ylabel("RMSE")
        rmse_ax.set_title(rmse_title, weight='bold')
        rmse_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}Bag Learner RMSE In vs Out Sample Results.png")

        plt.close("all")
        corr_title = f"{learner_name} Correlation vs. Leaf Size"
        corr_ax = bag_data["Correlation"]["In Sample"].mean(axis=1).plot(label="In Sample", grid=True, alpha=0.75)
        bag_data["Correlation"]["Out Sample"].mean(axis=1).plot(ax=corr_ax, label="Out Sample", grid=True, alpha=0.75)
        corr_ax.set_xlabel("Leaf Size")
        corr_ax.set_title(corr_title, weight='bold')
        corr_ax.set_ylabel("Correlation")
        corr_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}Bag Learner Correlation In vs Out Sample Results.png")
        # end region
    return


def get_experiment_three_charts(dt_data, rt_data, save_individual=True):
    # region Grouped Charts
    plt.close("all")
    plt.style.use('ggplot')
    fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    depth_title = "Tree Depth vs. Leaf Size"
    depth_ax = dt_data["Tree Depth"].mean(axis=1).plot(ax=axes1[0],
                                                       label="Decision Tree", grid=True, alpha=0.75, figsize=(10, 5))
    rt_data["Tree Depth"].mean(axis=1).plot(ax=depth_ax, label="Random Decision Tree", grid=True, alpha=0.75)
    depth_ax.set_xlabel("Leaf Size")
    depth_ax.set_ylabel("Tree Depth")
    depth_ax.set_title(depth_title, weight='bold')
    depth_ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
    depth_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")

    acc_title = "Accuracy vs. Leaf Size"
    acc_in_ax = dt_data["Metrics"]["In Sample Accuracy"].mean(axis=1).plot(ax=axes1[1],
                                                                           label="DT In Sample", grid=True, alpha=0.75,
                                                                           figsize=(10, 5))
    rt_data["Metrics"]["In Sample Accuracy"].mean(axis=1).plot(ax=acc_in_ax, label="RT In Sample", grid=True,
                                                               alpha=0.75)
    dt_data["Metrics"]["Out Sample Accuracy"].mean(axis=1).plot(ax=acc_in_ax, label="DT Out Sample", grid=True,
                                                                alpha=0.75)
    rt_data["Metrics"]["Out Sample Accuracy"].mean(axis=1).plot(ax=acc_in_ax, label="RT Out Sample", grid=True,
                                                                alpha=0.75)
    acc_in_ax.set_xlabel("Leaf Size")
    acc_in_ax.set_ylabel("Accuracy")
    acc_in_ax.set_title(acc_title, weight='bold')
    acc_in_ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
    acc_in_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}Experiment Three Depth and Accuracy Results.png")

    plt.close("all")
    fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    time_title = "Training Time vs. Leaf Size"
    time_ax = dt_data["Training Times"].mean(axis=1).plot(ax=axes2[0], label="Decision Tree", grid=True, alpha=0.75)
    rt_data["Training Times"].mean(axis=1).plot(ax=time_ax, label="Random Decision Tree", grid=True, alpha=0.75)
    time_ax.set_xlabel("Leaf Size")
    time_ax.set_title(time_title, weight='bold')
    time_ax.set_ylabel("Time (s)")
    time_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
    query_title = "Query Time vs. Leaf Size"
    query_ax = dt_data["Query Times"].mean(axis=1).plot(ax=axes2[1], label="Decision Tree", grid=True, alpha=0.75)
    rt_data["Query Times"].mean(axis=1).plot(ax=query_ax, label="Random Decision Tree", grid=True, alpha=0.75)
    query_ax.set_xlabel("Leaf Size")
    query_ax.set_title(query_title, weight='bold')
    query_ax.set_ylabel("Time (s)")
    query_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}Experiment Three Train and Query Time Results.png")

    plt.close("all")
    plt.style.use('ggplot')
    fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    acc_in_title = "In Sample Accuracy vs. Leaf Size"
    acc_in_ax = dt_data["Metrics"]["In Sample Accuracy"].mean(axis=1).plot(ax=axes1[0], label="DT In Sample",
                                                                           grid=True, alpha=0.75, figsize=(10, 5))
    rt_data["Metrics"]["In Sample Accuracy"].mean(axis=1).plot(ax=acc_in_ax, label="RT In Sample", grid=True,
                                                               alpha=0.75)
    acc_in_ax.set_xlabel("Leaf Size")
    acc_in_ax.set_ylabel("Metrics")
    acc_in_ax.set_title(acc_in_title, weight='bold')
    acc_in_ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
    acc_in_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
    acc_out_title = "Out Sample Accuracy vs. Leaf Size"
    acc_out_ax = dt_data["Metrics"]["Out Sample Accuracy"].mean(axis=1).plot(ax=axes1[1], label="DT Out Sample",
                                                                             grid=True, alpha=0.75, figsize=(10, 5))
    rt_data["Metrics"]["Out Sample Accuracy"].mean(axis=1).plot(ax=acc_out_ax, label="RT Out Sample", grid=True,
                                                                alpha=0.75)
    acc_out_ax.set_xlabel("Leaf Size")
    acc_out_ax.set_ylabel("Metrics")
    acc_out_ax.set_title(acc_out_title, weight='bold')
    acc_out_ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
    acc_out_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}Experiment Three Accuracy DT vs RT Results.png")

    plt.close("all")
    plt.style.use('ggplot')
    fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    var_in_title = "In Sample Variance vs. Leaf Size"
    var_in_ax = dt_data["Metrics"]["In Sample Variance"].mean(axis=1).plot(ax=axes1[0], label="DT In Sample",
                                                                           grid=True, alpha=0.75, figsize=(10, 5))
    rt_data["Metrics"]["In Sample Variance"].mean(axis=1).plot(ax=var_in_ax, label="RT In Sample", grid=True,
                                                               alpha=0.75)
    var_in_ax.set_xlabel("Leaf Size")
    var_in_ax.set_ylabel("Variance")
    var_in_ax.set_title(var_in_title, weight='bold')
    var_in_ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
    var_in_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
    var_out_title = "Out Sample Variance vs. Leaf Size"
    var_out_ax = dt_data["Metrics"]["Out Sample Variance"].mean(axis=1).plot(ax=axes1[1], label="DT Out Sample",
                                                                             grid=True, alpha=0.75, figsize=(10, 5))
    rt_data["Metrics"]["Out Sample Variance"].mean(axis=1).plot(ax=var_out_ax, label="RT Out Sample", grid=True,
                                                                alpha=0.75)
    var_out_ax.set_xlabel("Leaf Size")
    var_out_ax.set_ylabel("Variance")
    var_out_ax.set_title(var_out_title, weight='bold')
    var_out_ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
    var_out_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}Experiment Three Variance DT vs RT Results.png")

    plt.close("all")
    plt.style.use('ggplot')
    fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    std_in_title = "In Sample Standard Deviation vs. Leaf Size"
    std_in_ax = dt_data["Metrics"]["In Sample Std"].mean(axis=1).plot(ax=axes1[0], label="DT In Sample",
                                                                           grid=True, alpha=0.75, figsize=(10, 5))
    rt_data["Metrics"]["In Sample Std"].mean(axis=1).plot(ax=std_in_ax, label="RT In Sample", grid=True,
                                                               alpha=0.75)
    std_in_ax.set_xlabel("Leaf Size")
    std_in_ax.set_ylabel("Standard Deviation")
    std_in_ax.set_title(std_in_title, weight='bold')
    std_in_ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
    std_in_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
    std_out_title = "Out Sample Standard Deviation vs. Leaf Size"
    std_out_ax = dt_data["Metrics"]["Out Sample Std"].mean(axis=1).plot(ax=axes1[1], label="DT Out Sample",
                                                                             grid=True, alpha=0.75, figsize=(10, 5))
    rt_data["Metrics"]["Out Sample Std"].mean(axis=1).plot(ax=std_out_ax, label="RT Out Sample", grid=True,
                                                                alpha=0.75)
    std_out_ax.set_xlabel("Leaf Size")
    std_out_ax.set_ylabel("Standard Deviation")
    std_out_ax.set_title(std_out_title, weight='bold')
    std_out_ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
    std_out_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}Experiment Three Standard Deviation DT vs RT Results.png")


    plt.close("all")
    plt.style.use('ggplot')
    var_in_title = "Variance vs. Leaf Size"
    var_in_ax = dt_data["Metrics"]["In Sample Variance"].mean(axis=1).plot( label="DT In Sample",
                                                                           grid=True, alpha=0.75)
    rt_data["Metrics"]["In Sample Variance"].mean(axis=1).plot(ax=var_in_ax, label="RT In Sample", grid=True,
                                                               alpha=0.75)
    dt_data["Metrics"]["Out Sample Variance"].mean(axis=1).plot(ax=var_in_ax, label="DT Out Sample",
                                                                grid=True, alpha=0.5)
    rt_data["Metrics"]["Out Sample Variance"].mean(axis=1).plot(ax=var_in_ax, label="RT Out Sample", grid=True,
                                                                alpha=0.5)
    var_in_ax.set_xlabel("Leaf Size")
    var_in_ax.set_ylabel("Variance")
    var_in_ax.set_title(var_in_title, weight='bold')
    var_in_ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
    var_in_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}Experiment Three Combined Variance DT vs RT Results.png")

    # endregion

    if save_individual:
        # region Individual Charts

        plt.close("all")
        depth_title = "Tree Depth vs. Leaf Size"
        depth_ax = dt_data["Tree Depth"].mean(axis=1).plot(label="Decision Tree", grid=True, alpha=0.75)
        rt_data["Tree Depth"].mean(axis=1).plot(ax=depth_ax, label="Random Decision Tree", grid=True, alpha=0.75)
        depth_ax.set_xlabel("Leaf Size")
        depth_ax.set_ylabel("Tree Depth")
        depth_ax.set_title(depth_title, weight='bold')
        depth_ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
        depth_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}Experiment Three DT vs RT Tree Depth.png")

        plt.close("all")
        query_title = "Query Time vs. Leaf Size"
        query_ax = dt_data["Query Times"].mean(axis=1).plot(label="Decision Tree", grid=True, alpha=0.75)
        rt_data["Query Times"].mean(axis=1).plot(ax=query_ax, label="Random Decision Tree", grid=True, alpha=0.75)
        query_ax.set_xlabel("Leaf Size")
        query_ax.set_title(query_title, weight='bold')
        query_ax.set_ylabel("Time (s)")
        query_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}Experiment Three DT vs RT Query Time.png")

        plt.close("all")
        time_title = "Training Time vs. Leaf Size"
        time_ax = dt_data["Training Times"].mean(axis=1).plot(label="Decision Tree", grid=True, alpha=0.75)
        rt_data["Training Times"].mean(axis=1).plot(ax=time_ax, label="Random Decision Tree", grid=True, alpha=0.75)
        time_ax.set_xlabel("Leaf Size")
        time_ax.set_title(time_title, weight='bold')
        time_ax.set_ylabel("Time (s)")
        time_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}Experiment Three DT vs RT Training Time.png")

        plt.close("all")
        dt_rmse_title = "Decision Tree RMSE vs. Leaf Size"
        dt_rmse_ax = dt_data["RMSE"]["In Sample"].mean(axis=1).plot(label="RMSE In Sample", grid=True, alpha=0.75)
        dt_data["RMSE"]["Out Sample"].mean(axis=1).plot(ax=dt_rmse_ax, label="RMSE Out Sample", grid=True, alpha=0.75)
        dt_rmse_ax.set_xlabel("Leaf Size")
        dt_rmse_ax.set_title(dt_rmse_title, weight='bold')
        dt_rmse_ax.set_ylabel("RMSE")
        dt_rmse_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}DT RMSE In vs Out Sample.png")

        plt.close("all")
        dt_corr_title = "Decision Tree Correlation vs. Leaf Size"
        dt_corr_ax = dt_data["Correlation"]["In Sample"].mean(axis=1).plot(label="Corr. In Sample", grid=True,
                                                                           alpha=0.75)
        dt_data["Correlation"]["Out Sample"].mean(axis=1).plot(ax=dt_corr_ax, label="Corr. Out Sample", grid=True,
                                                               alpha=0.75)
        dt_corr_ax.set_xlabel("Leaf Size")
        dt_corr_ax.set_title(dt_corr_title, weight='bold')
        dt_corr_ax.set_ylabel("Correlation")
        dt_corr_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}DT Correlation In vs Out Sample.png")

        plt.close("all")
        rt_rmse_title = "Random Decision Tree RMSE vs. Leaf Size"
        rt_rmse_ax = rt_data["RMSE"]["In Sample"].mean(axis=1).plot(label="RMSE In Sample", grid=True, alpha=0.75)
        rt_data["RMSE"]["Out Sample"].mean(axis=1).plot(ax=rt_rmse_ax, label="RMSE Out Sample", grid=True, alpha=0.75)
        rt_rmse_ax.set_xlabel("Leaf Size")
        rt_rmse_ax.set_title(rt_rmse_title, weight='bold')
        rt_rmse_ax.set_ylabel("RMSE")
        rt_rmse_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}RT RMSE In vs Out Sample.png")

        plt.close("all")
        rt_corr_title = "Random Decision Tree Correlation vs. Leaf Size"
        rt_corr_ax = rt_data["Correlation"]["In Sample"].mean(axis=1).plot(label="Corr. In Sample", grid=True,
                                                                           alpha=0.75)
        rt_data["Correlation"]["Out Sample"].mean(axis=1).plot(ax=rt_corr_ax, label="Corr. Out Sample", grid=True,
                                                               alpha=0.75)
        rt_corr_ax.set_xlabel("Leaf Size")
        rt_corr_ax.set_title(rt_corr_title, weight='bold')
        rt_corr_ax.set_ylabel("Correlation")
        rt_corr_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}RT Correlation In vs Out Sample.png")

        plt.close("all")
        rt_acc_title = "Random Decision Tree Accuracy vs. Leaf Size"
        rt_acc_ax = rt_data["Metrics"]["In Sample Accuracy"].mean(axis=1).plot(label="Acc. In Sample", grid=True,
                                                                               alpha=0.75)
        rt_data["Metrics"]["Out Sample Accuracy"].mean(axis=1).plot(ax=rt_acc_ax, label="Acc. Out Sample", grid=True,
                                                                    alpha=0.75)
        rt_acc_ax.set_xlabel("Leaf Size")
        rt_acc_ax.set_title(rt_acc_title, weight='bold')
        rt_acc_ax.set_ylabel("Accuracy")
        rt_acc_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}RT Accuracy In vs Out Sample.png")

        plt.close("all")
        dt_acc_title = "Decision Tree Accuracy vs. Leaf Size"
        dt_acc_ax = rt_data["Metrics"]["In Sample Accuracy"].mean(axis=1).plot(label="Acc. In Sample", grid=True,
                                                                               alpha=0.75)
        dt_data["Metrics"]["Out Sample Accuracy"].mean(axis=1).plot(ax=dt_acc_ax, label="Acc. Out Sample", grid=True,
                                                                    alpha=0.75)
        dt_acc_ax.set_xlabel("Leaf Size")
        dt_acc_ax.set_title(dt_acc_title, weight='bold')
        dt_acc_ax.set_ylabel("Accuracy")
        dt_acc_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}DT Accuracy In vs Out Sample.png")

        plt.close("all")
        plt.style.use('ggplot')
        std_in_title = "In Sample Standard Deviation vs. Leaf Size"
        std_in_ax = dt_data["Metrics"]["In Sample Std"].mean(axis=1).plot(label="DT In Sample",
                                                                          grid=True, alpha=0.75)
        rt_data["Metrics"]["In Sample Std"].mean(axis=1).plot(ax=std_in_ax, label="RT In Sample", grid=True,
                                                              alpha=0.75)
        std_in_ax.set_xlabel("Leaf Size")
        std_in_ax.set_ylabel("Standard Deviation")
        std_in_ax.set_title(std_in_title, weight='bold')
        std_in_ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
        std_in_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}Standard Deviation In Sample DT vs RT Results.png")

        plt.close("all")
        plt.style.use('ggplot')
        std_out_title = "Out Sample Standard Deviation vs. Leaf Size"
        std_out_ax = dt_data["Metrics"]["Out Sample Std"].mean(axis=1).plot(label="DT Out Sample",
                                                                            grid=True, alpha=0.75)
        rt_data["Metrics"]["Out Sample Std"].mean(axis=1).plot(ax=std_out_ax, label="RT Out Sample", grid=True,
                                                               alpha=0.75)
        std_out_ax.set_xlabel("Leaf Size")
        std_out_ax.set_ylabel("Standard Deviation")
        std_out_ax.set_title(std_out_title, weight='bold')
        std_out_ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
        std_out_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}Standard Deviation Out Sample DT vs RT Results.png")

        plt.close("all")
        plt.style.use('ggplot')
        var_in_title = "In Sample Variance vs. Leaf Size"
        var_in_ax = dt_data["Metrics"]["In Sample Variance"].mean(axis=1).plot(label="DT In Sample",
                                                                               grid=True, alpha=0.75)
        rt_data["Metrics"]["In Sample Variance"].mean(axis=1).plot(ax=var_in_ax, label="RT In Sample", grid=True,
                                                                   alpha=0.75)
        var_in_ax.set_xlabel("Leaf Size")
        var_in_ax.set_ylabel("Variance")
        var_in_ax.set_title(var_in_title, weight='bold')
        var_in_ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
        var_in_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}Variance In Sample DT vs RT Results.png")

        plt.close("all")
        plt.style.use('ggplot')
        var_out_title = "Out Sample Variance vs. Leaf Size"
        var_out_ax = dt_data["Metrics"]["Out Sample Variance"].mean(axis=1).plot(label="DT Out Sample",
                                                                                 grid=True, alpha=0.75)
        rt_data["Metrics"]["Out Sample Variance"].mean(axis=1).plot(ax=var_out_ax, label="RT Out Sample", grid=True,
                                                                    alpha=0.75)
        var_out_ax.set_xlabel("Leaf Size")
        var_out_ax.set_ylabel("Variance")
        var_out_ax.set_title(var_out_title, weight='bold')
        var_out_ax.legend(markerscale=1.1, frameon=True, edgecolor="black")
        var_out_ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}Variance Out Sample DT vs RT Results.png")
        # endregion
    return


def get_training_times(data, learner, subsets, cv=5, bag_learn=False, **kwargs):
    is_tree = False
    if learner.__name__ == "DTLearner" or learner.__name__ == "RTLearner":
        is_tree = True
    if bag_learn:
        learner = np.array([learner(learner=dt.DTLearner, **kwargs) for _ in range(cv)])
    else:
        learner = np.array([learner(**kwargs) for _ in range(cv)])
    times = np.zeros(shape=(cv,))
    depth = np.zeros(shape=(cv,))
    for i in range(cv):
        temp_data = data[subsets[i], :]
        x = temp_data[:, :-1]
        y = temp_data[:, -1]
        start_time = time.time()
        learner[i].add_evidence(data_x=x, data_y=y)
        end_time = time.time()
        if is_tree:
            depth[i] = np.nanmax(np.nanmax(learner[i].tree, axis=0))
        elapsed_time = end_time - start_time
        times[i] = elapsed_time
    return times, learner, depth


def get_query_stats(data, learners, subsets, cv=5):
    times = np.zeros(shape=(cv,))
    for i in range(cv):
        temp_data = data[subsets[i], :]
        x = temp_data[:, :-1]
        start_time = time.time()
        y_pred = learners[i].query(points=x)
        end_time = time.time()
        elapsed_time = end_time - start_time
        times[i] = elapsed_time
    return times


def get_rmse_corr(train_data, test_data, train_subsets, learners, cv=5, is_bag=False):
    corr_in = np.zeros(shape=(cv,))
    corr_out = np.zeros(shape=(cv,))
    rmse_in = np.zeros(shape=(cv,))
    rmse_out = np.zeros(shape=(cv,))
    if not is_bag:
        acc_in = np.zeros(shape=(cv,))
        acc_out = np.zeros(shape=(cv,))
        var_in = np.zeros(shape=(cv,))
        var_out = np.zeros(shape=(cv,))
        std_in = np.zeros(shape=(cv,))
        std_out = np.zeros(shape=(cv,))
    for i in range(cv):
        temp_train_data = train_data[train_subsets[i], :]
        train_x = temp_train_data[:, :-1]
        train_y = temp_train_data[:, -1]

        test_x = test_data[:, :-1]
        test_y = test_data[:, -1]

        y_pred_in = learners[i].query(points=train_x)
        y_pred_out = learners[i].query(points=test_x)

        if not is_bag:
            check_in = np.asarray(np.isclose(y_pred_in, train_y, atol=0.01), dtype=np.int32)
            check_out = np.asarray(np.isclose(y_pred_out, test_y, atol=0.01), dtype=np.int32)
            acc_in[i] = check_in.sum() / train_y.shape[0]
            acc_out[i] = check_out.sum() / test_y.shape[0]
            var_in[i] = np.nanvar(y_pred_in)
            var_out[i] = np.nanvar(y_pred_out)
            std_in[i] = np.nanstd(y_pred_in)
            std_out[i] = np.nanstd(y_pred_out)

        corr_in[i] = np.corrcoef(y_pred_in, y=train_y)[0, 1]
        rmse_in[i] = np.sqrt(((train_y - y_pred_in) ** 2).sum() / train_y.shape[0])
        corr_out[i] = np.corrcoef(y_pred_out, y=test_y)[0, 1]
        rmse_out[i] = np.sqrt(((test_y - y_pred_out) ** 2).sum() / train_y.shape[0])

    if is_bag:
        return corr_in, corr_out, rmse_in, rmse_out
    else:
        return corr_in, corr_out, rmse_in, rmse_out, acc_in, acc_out, var_in, var_out, std_in, std_out


fig_dir = "figures/"

if __name__ == "__main__":
    import InsaneLearner as insane
    
    
    np.random.seed(903475599)
    CHECK_FOLDER = os.path.isdir(fig_dir)
    if not CHECK_FOLDER:
        os.makedirs(fig_dir)

    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)

    file_name = sys.argv[1]
    if len(file_name) < 3:
        file_name = "Istanbul.csv"
    else:
        split_names = file_name.split("/")
        file_name = split_names[-1]

    with util.get_learner_data_file(file_name) as f:
        alldata = np.genfromtxt(f, delimiter=",")
        # Skip the date column and header row if we're working on Istanbul data
        if file_name == "Istanbul.csv":
            alldata = alldata[1:, 1:]

        datasize = alldata.shape[0]
        cutoff = int(datasize * 0.75)
        permutation = np.random.permutation(alldata.shape[0])
        col_permutation = np.random.permutation(alldata.shape[1] - 1)
        train_data = alldata[permutation[:cutoff], :]
        # train_x = train_data[:,:-1]
        train_x = train_data[:, col_permutation]
        train_y = train_data[:, -1]
        test_data = alldata[permutation[cutoff:], :]
        # test_x = test_data[:,:-1]
        test_x = test_data[:, col_permutation]
        test_y = test_data[:, -1]

    num_leaves = 31
    bag_leaves = 31
    run_all = True
    st = 0
    end = 0
    elap = 0
    cv = 10
    bag_cv = 2
    cols = [f"cv {i}" for i in range(cv)]
    bag_cols = [f"cv {i}" for i in range(bag_cv)]

    Results = \
        {
            "Decision Tree Results":
                {
                    "Training Times": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols),
                    "Query Times": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols),
                    "RMSE":
                        {
                            "In Sample": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols),
                            "Out Sample": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols)
                        },
                    "Correlation":
                        {
                            "In Sample": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols),
                            "Out Sample": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols)
                        },
                    "Metrics":
                        {"In Sample Accuracy": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols),
                         "Out Sample Accuracy": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols),
                         "In Sample Variance": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols),
                         "Out Sample Variance": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols),
                         "In Sample Std": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)),
                                                                      columns=cols),
                         "Out Sample Std": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)),
                                                                       columns=cols)
                         },
                    "Tree Depth": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols)
                },
            "Random Decision Tree Results":
                {
                    "Training Times": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols),
                    "Query Times": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols),
                    "RMSE":
                        {"In Sample": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols),
                         "Out Sample": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols)
                         },
                    "Correlation":
                        {"In Sample": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols),
                         "Out Sample": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols)},
                    "Metrics":
                        {"In Sample Accuracy": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols),
                         "Out Sample Accuracy": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols),
                         "In Sample Variance": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols),
                         "Out Sample Variance": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols),
                         "In Sample Std": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)),
                                                                      columns=cols),
                         "Out Sample Std": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)),
                                                                       columns=cols)
                         },
                    "Tree Depth": pd.DataFrame(data=np.zeros(shape=(num_leaves, cv)), columns=cols)
                },
            "Bag Learner Results":
                {
                    "Training Times": pd.DataFrame(data=np.zeros(shape=(num_leaves, bag_cv)), columns=bag_cols),
                    "Query Times": pd.DataFrame(data=np.zeros(shape=(num_leaves, bag_cv)), columns=bag_cols),
                    "RMSE":
                        {
                            "In Sample": pd.DataFrame(data=np.zeros(shape=(num_leaves, bag_cv)), columns=bag_cols),
                            "Out Sample": pd.DataFrame(data=np.zeros(shape=(num_leaves, bag_cv)), columns=bag_cols)
                        },
                    "Correlation":
                        {
                            "In Sample": pd.DataFrame(data=np.zeros(shape=(num_leaves, bag_cv)), columns=bag_cols),
                            "Out Sample": pd.DataFrame(data=np.zeros(shape=(num_leaves, bag_cv)), columns=bag_cols)
                        },
                    "Metrics":
                        {
                            "In Sample Accuracy": pd.DataFrame(data=np.zeros(shape=(num_leaves, bag_cv)),
                                                               columns=bag_cols),
                            "Out Sample Accuracy": pd.DataFrame(data=np.zeros(shape=(num_leaves, bag_cv)),
                                                                columns=bag_cols),
                            "In Sample Variance": pd.DataFrame(data=np.zeros(shape=(num_leaves, bag_cv)),
                                                               columns=bag_cols),
                            "Out Sample Variance": pd.DataFrame(data=np.zeros(shape=(num_leaves, bag_cv)),
                                                                columns=bag_cols),
                            "In Sample Std": pd.DataFrame(data=np.zeros(shape=(num_leaves, bag_cv)),
                                                                         columns=bag_cols),
                            "Out Sample Std": pd.DataFrame(data=np.zeros(shape=(num_leaves, bag_cv)),
                                                                          columns=bag_cols)
                        }
                }
        }
    learners = np.array([dt.DTLearner, rt.RTLearner, bg.BagLearner])
    training_subsets = np.random.choice(np.arange(train_data.shape[0]), size=(cv, train_data.shape[0]),
                                        replace=True)
    total_start_time = time.time()
    print("Starting Evaluations")
    if run_all:
        dt_start_time = time.time()
        for i in range(1, num_leaves + 1):
            print(f"\tCurrent Leaf Size: {i}")
            kwargs = {"leaf_size": i}

            dt_training_times, dt_learners, dt_depth = get_training_times(learner=dt.DTLearner, data=train_data,
                                                                          subsets=training_subsets, kwargs=kwargs,
                                                                          cv=cv)
            dt_query_times = get_query_stats(data=train_data, learners=dt_learners, subsets=training_subsets, cv=cv)
            dt_corr_in, dt_corr_out, dt_rmse_in, \
            dt_rmse_out, dt_acc_in, dt_acc_out, dt_var_in, \
            dt_var_out, dt_std_in, dt_std_out = get_rmse_corr(train_data=train_data, test_data=test_data,
                                                               train_subsets=training_subsets,
                                                               learners=dt_learners, cv=cv)
            Results["Decision Tree Results"]["Training Times"].iloc[i - 1, :] = 1
            b = Results["Decision Tree Results"]["Training Times"].iloc[i - 1, :]
            Results["Decision Tree Results"]["Training Times"].iloc[i - 1, :] = dt_training_times
            Results["Decision Tree Results"]["Query Times"].iloc[i - 1, :] = dt_query_times
            Results["Decision Tree Results"]["Tree Depth"].iloc[i - 1, :] = dt_depth
            Results["Decision Tree Results"]["RMSE"]["In Sample"].iloc[i - 1, :] = dt_rmse_in
            Results["Decision Tree Results"]["RMSE"]["Out Sample"].iloc[i - 1, :] = dt_rmse_out
            Results["Decision Tree Results"]["Correlation"]["In Sample"].iloc[i - 1, :] = dt_corr_in
            Results["Decision Tree Results"]["Correlation"]["Out Sample"].iloc[i - 1, :] = dt_corr_out
            Results["Decision Tree Results"]["Metrics"]["In Sample Accuracy"].iloc[i - 1, :] = dt_acc_in
            Results["Decision Tree Results"]["Metrics"]["Out Sample Accuracy"].iloc[i - 1, :] = dt_acc_out
            Results["Decision Tree Results"]["Metrics"]["In Sample Variance"].iloc[i - 1, :] = dt_var_in
            Results["Decision Tree Results"]["Metrics"]["Out Sample Variance"].iloc[i - 1, :] = dt_var_out
            Results["Decision Tree Results"]["Metrics"]["In Sample Std"].iloc[i - 1, :] = dt_std_in
            Results["Decision Tree Results"]["Metrics"]["Out Sample Std"].iloc[i - 1, :] = dt_std_out

            rt_training_times, rt_learners, rt_depth = get_training_times(learner=rt.RTLearner, data=train_data,
                                                                          subsets=training_subsets, kwargs=kwargs,
                                                                          cv=cv)

            rt_query_times = get_query_stats(data=train_data, learners=rt_learners, subsets=training_subsets, cv=cv)

            rt_corr_in, rt_corr_out, rt_rmse_in, \
            rt_rmse_out, rt_acc_in, rt_acc_out, rt_var_in, \
            rt_var_out, rt_std_in, rt_std_out = get_rmse_corr(train_data=train_data, test_data=test_data,
                                                               train_subsets=training_subsets,
                                                               learners=rt_learners, cv=cv)

            Results["Random Decision Tree Results"]["Training Times"].iloc[i - 1, :] = rt_training_times
            Results["Random Decision Tree Results"]["Query Times"].iloc[i - 1, :] = rt_query_times
            Results["Random Decision Tree Results"]["Tree Depth"].iloc[i - 1, :] = rt_depth
            Results["Random Decision Tree Results"]["RMSE"]["In Sample"].iloc[i - 1, :] = rt_rmse_in
            Results["Random Decision Tree Results"]["RMSE"]["Out Sample"].iloc[i - 1, :] = rt_rmse_out
            Results["Random Decision Tree Results"]["Correlation"]["In Sample"].iloc[i - 1, :] = rt_corr_in
            Results["Random Decision Tree Results"]["Correlation"]["Out Sample"].iloc[i - 1, :] = rt_corr_out
            Results["Random Decision Tree Results"]["Metrics"]["In Sample Accuracy"].iloc[i - 1, :] = rt_acc_in
            Results["Random Decision Tree Results"]["Metrics"]["Out Sample Accuracy"].iloc[i - 1, :] = rt_acc_out
            Results["Random Decision Tree Results"]["Metrics"]["In Sample Variance"].iloc[i - 1, :] = rt_var_in
            Results["Random Decision Tree Results"]["Metrics"]["Out Sample Variance"].iloc[i - 1, :] = rt_var_out
            Results["Random Decision Tree Results"]["Metrics"]["In Sample Std"].iloc[i - 1, :] = rt_std_in
            Results["Random Decision Tree Results"]["Metrics"]["Out Sample Std"].iloc[i - 1, :] = rt_std_out

        dt_end_time = time.time()
        dt_elapsed_time = dt_end_time - dt_start_time
        print(f"DT and RT Total Elapsed Time: {dt_elapsed_time}")

        get_experiment_one_charts(data=Results["Decision Tree Results"])

        get_experiment_three_charts(dt_data=Results["Decision Tree Results"],
                                    rt_data=Results["Random Decision Tree Results"])

    bag_start_time = time.time()
    print("Evaluation of Bag Learner:")
    for j in range(1, bag_leaves + 1):
        print(f"\tCurrent Leaf Size: {j}")
        kwargs = {"leaf_size": j}

        bag_training_times, bag_learners, _ = get_training_times(learner=bg.BagLearner, data=train_data,
                                                                 subsets=training_subsets, bag_learn=True,
                                                                 kwargs=kwargs, cv=bag_cv)

        bag_query_times = get_query_stats(data=train_data, learners=bag_learners, subsets=training_subsets,
                                          cv=bag_cv)

        bag_corr_in, bag_corr_out, bag_rmse_in, \
        bag_rmse_out = get_rmse_corr(train_data=train_data, test_data=test_data, train_subsets=training_subsets,
                                     learners=bag_learners, cv=bag_cv, is_bag=True)

        Results["Bag Learner Results"]["Training Times"].iloc[j - 1, :] = bag_training_times
        Results["Bag Learner Results"]["Query Times"].iloc[j - 1, :] = bag_query_times
        Results["Bag Learner Results"]["RMSE"]["In Sample"].iloc[j - 1, :] = bag_rmse_in
        Results["Bag Learner Results"]["RMSE"]["Out Sample"].iloc[j - 1, :] = bag_rmse_out
        Results["Bag Learner Results"]["Correlation"]["In Sample"].iloc[j - 1, :] = bag_corr_in
        Results["Bag Learner Results"]["Correlation"]["Out Sample"].iloc[j - 1, :] = bag_corr_out


    bag_end_time = time.time()
    bag_elapsed = bag_end_time - bag_start_time
    print(f"Bag Learner Total Elapsed Time: {bag_elapsed}\n")

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"Total Elapsed Time: {total_elapsed_time}")

    get_experiment_two_charts(bag_data=Results["Bag Learner Results"])
    exit()
