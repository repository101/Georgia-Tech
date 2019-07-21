""" 			  		 			 	 	 		 		 	  		   	  			  	
Template for implementing StrategyLearner  (c) 2016 Tucker Balch 			  		 			 	 	 		 		 	  		   	  			  	

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

import datetime as dt
import pandas as pd
import util as ut
import numpy as np
import random
import os
import sys

import QLearner as ql
import marketsimcode as mktsim


class StrategyLearner(object):
	# constructor
	def __init__(self, verbose=False, impact=0.0):
		pd.reset_option('all')
		self.verbose = verbose
		self.discretization_bins = 10
		self.impact = impact
		self.SimpleMovingAverageBins = None
		self.SimpleMovingAverageRatioBins = None
		self.RollingStandardDeviationBins = None
		self.RelativeStrengthIndexBins = None
		self.BollingerBandPercentageBins = None
		self.MACD_CrossOverBins = None
		self.MomentumBins = None
		self.holdings = 0
		self.max_epochs = 200
		self.convergence = False
		self.learner = None
		self.number_of_states = None
		self.previous_dataframe = None
	
	def author(self):
		return "jadams334"
	
	def discretize(self, dataframe, indicator):
		# Convert continuous indicators into integers
		try:
			discretized_indicator, indicator_bins = pd.qcut(dataframe, self.discretization_bins, retbins=True, labels=False)
	
			if indicator == 0:
				# Simple Moving Average
				self.SimpleMovingAverageBins = indicator_bins
			if indicator == 1:
				# Simple Moving Average Ratio
				self.SimpleMovingAverageRatioBins = indicator_bins
			if indicator == 2:
				# Rolling Standard Deviation
				self.RollingStandardDeviationBins = indicator_bins
			if indicator == 3:
				# Relative Strength Index
				self.RelativeStrengthIndexBins = indicator_bins
			if indicator == 4:
				# Bollinger Bands Percentage
				self.BollingerBandPercentageBins = indicator_bins
			if indicator == 5:
				# MACD Crossover
				self.MACD_CrossOverBins = indicator_bins
			if indicator == 6:
				# Momentum
				self.MomentumBins = indicator_bins
				
			return discretized_indicator, indicator_bins
		except Exception as err:
			if self.verbose:
				print "Failed in discretize"
				print err
	
	def get_states(self, dataframe):
		try:
			states = dataframe.sum(axis=1)
			return states
		except Exception as err:
			if self.verbose:
				print "Failed in get states"
				print err
	
	def get_reward(self, closePrice, add_impact=False):
		# Reward is best as percentage change
		reward = self.holdings * closePrice
		if add_impact:
			reward *= (1 - self.impact)
			return reward
		return reward
	
	def get_indicators(self, pricesDataFrame, fillNa=False, otherFill=False):
		try:
			lookback=14
			discretized_indicators_dataframe = pd.DataFrame()
			# Copying prices to result as to separate the dataframes and not taint data
			resultDataFrame = pricesDataFrame.copy()
	
			#region Simple Moving Average
			SimpleMovingAverage = mktsim.getRollingMean(resultDataFrame, window=lookback)
			discretized_SimpleMovingAverage, discretized_SimpleMovingAverage_Bins = self.discretize(SimpleMovingAverage, 0)
			if fillNa:
				discretized_SimpleMovingAverage.fillna(10, inplace=True)
			discretized_indicators_dataframe["Simple Moving Average"] = discretized_SimpleMovingAverage
			#endregion
			
			#region Simple Moving Average Ratio
			SimpleMovingAverageRatio = mktsim.PriceSMA_Ratio(resultDataFrame, SimpleMovingAverage)
			discretized_SimpleMovingAverageRatio, discretized_SimpleMovingAverageRatio_Bins = self.discretize(SimpleMovingAverageRatio, 1)
			if fillNa:
				discretized_SimpleMovingAverageRatio.fillna(10, inplace=True)
			discretized_indicators_dataframe["Simple Moving Average Ratio"] = discretized_SimpleMovingAverageRatio
			#endregion
			
			#region Rolling Standard Deviation
			RollingStandardDeviation = mktsim.getRollingStandardDeviation(resultDataFrame['Adj Close'], window=lookback)
			discretized_RollingStandardDeviation, discretized_RollingStandardDeviation_Bins = self.discretize(RollingStandardDeviation, 2)
			if fillNa:
				discretized_RollingStandardDeviation.fillna(10, inplace=True)
			discretized_indicators_dataframe["Rolling Standard Deviation"] = discretized_RollingStandardDeviation
			#endregion
			
			#region Relative Strength Index
			RelativeStrengthIndex = mktsim.Relative_Strength_Index(resultDataFrame)
			discretized_RelativeStrengthIndex, discretized_RelativeStrengthIndex_Bins = self.discretize(RelativeStrengthIndex, 3)
			if fillNa:
				discretized_RelativeStrengthIndex.fillna(10, inplace=True)
			discretized_indicators_dataframe["Relative Strength Index"] = discretized_RelativeStrengthIndex
			#endregion
			
			#region BollingerBands Percentage
			UpperBand, LowerBand = mktsim.getBollingerBands(SimpleMovingAverage, RollingStandardDeviation)
			BollingerBandPercentage = mktsim.BollingerBand_Ratio(resultDataFrame, SimpleMovingAverage, RollingStandardDeviation)
			discretized_BollingerBandPercentage, discretized_BollingerBandPercentage_Bins = self.discretize(BollingerBandPercentage, 4)
			if fillNa:
				discretized_BollingerBandPercentage.fillna(10, inplace=True)
			self.BollingerBandPercentageBins = discretized_BollingerBandPercentage_Bins
			discretized_indicators_dataframe["Bollinger Band Percentage"] = discretized_BollingerBandPercentage
			#endregion
			
			#region MACD
			MACD_CrossOver = mktsim.getMomentum(resultDataFrame)
			discretized_MACD_CrossOver, discretized_MACD_CrossOver_Bins = self.discretize(MACD_CrossOver, 5)
			if fillNa:
				discretized_MACD_CrossOver.fillna(10.0, inplace=True)
			discretized_indicators_dataframe["MACD CrossOver"] = discretized_MACD_CrossOver * 1.0
			#endregion
			
			#region Momentum
			Momentum = resultDataFrame['DailyMomentum']
			discretized_Momentum_CrossOver, discretized_Momentum_Bins = self.discretize(Momentum, 6)
			if fillNa:
				discretized_Momentum_CrossOver.fillna(10.0, inplace=True)
			discretized_indicators_dataframe["Daily Momentum"] = discretized_Momentum_CrossOver
			#endregion
			if otherFill:
				discretized_indicators_dataframe.ffill(inplace=True)
				discretized_indicators_dataframe.bfill(inplace=True)
				
			#region Daily Returns
			daily_returns = mktsim.get_daily_returns(resultDataFrame['Adj Close'])
			resultDataFrame["DailyReturns"] = daily_returns
			#endregion
			return resultDataFrame, discretized_indicators_dataframe, daily_returns
		except Exception as err:
			if self.verbose:
				print "Failed during get_indicators"
				print err
	
	# this method should create a QLearner, and train it for trading
	def addEvidence(self, symbol="IBM", \
	                sd=dt.datetime(2008, 1, 1), \
	                ed=dt.datetime(2009, 1, 1), \
	                sv=10000):
		try:
			
			# add your code to do learning here.
	
			
			# example usage of the old backward compatible util function
			syms = [symbol]
			dates = pd.date_range(sd, ed)
			prices_all = ut.get_data(syms, dates)  # automatically adds SPY
			prices = prices_all[syms]  # only portfolio symbols
			prices_SPY = prices_all['SPY']  # only SPY, for comparison later
			if self.verbose: print prices
			
			# example use with new colname
			volume_all = ut.get_data(syms, dates,
			                         colname="Volume")  # automatically adds SPY
			volume = volume_all[syms]  # only portfolio symbols
			volume_SPY = volume_all['SPY']  # only SPY, for comparison later
			if self.verbose: print volume
			prices.rename(columns={symbol: "Adj Close"}, inplace=True)
			
			#region Create Q_Learner
			try:
				self.number_of_states = prices.shape[0]
			except Exception as err:
				if self.verbose:
					print "Failed when attempting to set number of states"
					print err
			number_of_actions = 3
			if self.number_of_states is None:
				self.number_of_states = len(prices)
			
			# So we can reference as needed
			self.learner = ql.QLearner(num_states=self.number_of_states, \
			                      num_actions=number_of_actions, \
			                      alpha=0.2, \
			                      gamma=0.9, \
			                      rar=0.98, \
			                      radr=0.999, \
			                      dyna=0, \
			                      verbose=False, actions=[0, 1, 2])
			#endregion
			
			
			#region Train Q_Learner
			
			#region Get Indicators and Discretize
			FullDataFrame, discretizedDataFrame, daily_returns = self.get_indicators(prices, fillNa=False, otherFill=True)
			FullDataFrame["States"] = self.get_states(discretizedDataFrame[["Bollinger Band Percentage", "Relative Strength Index", "Daily Momentum"]])
			#endregion
			
			#region Training
			epoch = 0
			tradeDataFrame = pd.DataFrame(data=np.zeros(shape=len(FullDataFrame)), index=FullDataFrame.index)
			while(epoch <= self.max_epochs) & (self.convergence == False):
				# Holdings X daily_returns will give us our reward
				holdings = self.holdings
				action = self.learner.querysetstate(FullDataFrame["States"][0])
				for index, row in FullDataFrame.iterrows():
					reward = self.get_reward((daily_returns.loc[index]))
					if action in [1, 2]:
						# ACTIONS
						#   0 = Nothing
						#   1 = Short
						#   2 = Long
						# Apply impact
						reward = self.get_reward((daily_returns.loc[index]), add_impact=True)
					print ""
					try:
						action = self.learner.query(row["States"], reward)
						print ""
					except Exception as err:
						if self.verbose:
							print "Error occurred when attempting to query the action"
							exc_type, exc_obj, exc_tb = sys.exc_info()
							print exc_obj
							print exc_tb.tb_lineno
							print err
					if action == 0:
						# Do Nothing
						tradeDataFrame.loc[index] = 0
						print ""
					elif action == 1:
						# SHORT ~~ Sell
						# Check holdings
						if self.holdings == 0:
							tradeDataFrame.loc[index] = -1000
						elif self.holdings == 1000:
							# TODO: Add ability to buy 1000 or 2000
							tradeDataFrame.loc[index] = -2000
						elif self.holdings == -1000:
							# TODO: Add ability to sell 1000 or 2000
							tradeDataFrame.loc[index] = 0
						else:
							if self.verbose:
								print "Unexpected value of holdings occurred"
					elif action == 2:
						# LONG ~~ Buy
						# Check holdings
						if self.holdings == 0:
							# Holdings are 0 we can buy 1000
							tradeDataFrame.loc[index] = 1000
						elif self.holdings == 1000:
							# holdings are 1000 we can sell 1000 or 2000
							# TODO: Add ability to sell 1000 or 2000
							tradeDataFrame.loc[index] = 0
						elif self.holdings == -1000:
							# TODO: Add ability to buy 1000 or 2000
							tradeDataFrame.loc[index] = 2000
						else:
							if self.verbose:
								print "Unexpected value of holdings occurred"
					else:
						if self.verbose:
							print "Something happened while training and an invalid action was returned"
					self.holdings = self.holdings + tradeDataFrame.loc[index][0]
				# Compare results of self.previous_dataframe with new dataframe to see if we have converged
				if (self.previous_dataframe is None):
					pass
				else:
					# We compare the new dataframe to the previous dataframe
					
					print ""
				# Set previous dataframe to the current because we will use that in the next comparison
				self.previous_dataframe = tradeDataFrame
			#endregion
			
			#endregion
			
			return
		except Exception as AddEvidenceException:
			if self.verbose:
				print "Error occurred when attempting to Train the learner"
				exc_type, exc_obj, exc_tb = sys.exc_info()
				print exc_obj
				print exc_tb.tb_lineno
				print AddEvidenceException

	
	# this method should use the existing policy and test it against new data
	def testPolicy(self, symbol="IBM", \
	               sd=dt.datetime(2009, 1, 1), \
	               ed=dt.datetime(2010, 1, 1), \
	               sv=10000):
		try:
			# here we build a fake set of trades
			# your code should return the same sort of data
			dates = pd.date_range(sd, ed)
			prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
			trades = prices_all[[symbol, ]]  # only portfolio symbols
			trades_SPY = prices_all['SPY']  # only SPY, for comparison later
			trades.values[:, :] = 0  # set them all to nothing
			trades.values[0, :] = 1000  # add a BUY at the start
			trades.values[40, :] = -1000  # add a SELL
			trades.values[41, :] = 1000  # add a BUY
			trades.values[60, :] = -2000  # go short from long
			trades.values[61, :] = 2000  # go long from short
			trades.values[-1, :] = -1000  # exit on the last day
			if self.verbose: print type(trades)  # it better be a DataFrame!
			if self.verbose: print trades
			if self.verbose: print prices_all
			return trades
		except Exception as err:
			print "Failed during test policy"
			print err


if __name__ == "__main__":
	tester = StrategyLearner(verbose=True, impact=0.000)
	tester.addEvidence(symbol='AAPL', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
	print "One does not simply think up a strategy"
