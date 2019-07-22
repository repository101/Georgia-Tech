""" 			  		 			 	 	 		 		 	  		   	  			  	
Template for implementing QLearner  (c) 2015 Tucker Balch 			  		 			 	 	 		 		 	  		   	  			  	

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
import random as rand 			  		 			 	 	 		 		 	  		   	  			  	


class QLearner(object): 			  		 			 	 	 		 		 	  		   	  			  	
	
	def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False):
		self.verbose = verbose
		self.s = 0
		self.a = 0
		self.num_actions = num_actions  # Number of actions available
		self.num_states = num_states    # Number of states to consider
		self.alpha = alpha  # The learning rate
		self.gamma = gamma  # The discount rate
		self.rar = rar  # Random action rate
		# To update rar, self.rar = self.rar * self.radr
		self.radr = radr    # Random action decay rate
		self.dyna = dyna    # Number of dyna updates, typically 200
		# self.q_table = np.random.uniform(-1.0, 1.0, size=(self.num_states, self.num_actions))
		self.q_table = np.zeros((self.num_states, self.num_actions))
		self.action_meaning = {0: "Move North",
		                       1: "Move East",
		                       2: "Move South",
		                       3: "Move West"}
		
	def author(self):
		return "jadams334"
	
	def querysetstate(self, s, random=True):
		"""
		@summary: Update the state without updating the Q-table
		@param s: The new state
		@returns: The selected action
		"""
		# Action Tuple  ---  < S, A, S_Prime, R >
		# Action Tuple  ---  < Current_State, Action_taken, New_State, Reward >
		if random:
			if self.perform_random_action_or_not():
				# We get a random action
				# Take that action
				action = np.random.choice([0, 1, 2, 3])
				return action
		else:
			action = np.argmax(self.q_table[s])
		if self.verbose:
			print "s =", s, "a =", action
		self.s = s
		self.a = action
		return action
		
	def query(self, s_prime, r):
		"""
		@summary: Update the Q table and return an action
		@param s_prime: The new state
		@param r: The new state
		@returns: The selected action
		"""
		# Action Tuple  ---  < S, A, S_Prime, R >
		# Action Tuple  ---  < Current_State, Action_taken, New_State, Reward >
		# Update the rule
		# https://classroom.udacity.com/courses/ud501/lessons/5247432317/concepts/53538285920923
		self.update_q_table(s=self.s, a=self.a, s_prime=s_prime, r=r)
		action = self.querysetstate(s_prime)
		self.update_rar()
		if self.verbose:
			print "s =", s_prime, "a =", action, "r =", r
		return action
		
	def perform_random_action_or_not(self):
		if self.verbose:
			print "Current random action rate is {}".format(self.rar)
			print "Chance of taking a random action is {}".format(self.rar)
			print "Chance of not taking a random action is {}".format(1.0 - self.rar)
		return np.random.uniform(0.0, 1.0) <= self.rar
	
	def update_rar(self):
		old_rar = self.rar
		new_rar = self.rar * self.radr
		self.rar = new_rar
		if self.verbose:
			print "Rar has been updated from {} to {}".format(old_rar, new_rar)
		return
	
	def update_q_table(self, s=0, a=0, s_prime=0, r=0):
		Part_1 = (1 - self.alpha) * self.q_table[s, a]      # Correct
		# Later_Rewards = Q[s_prime, argmax(Q[s_prime, a_prime)]
		later_rewards = self.q_table[s_prime, np.argmax(self.q_table[s_prime])]     # Correct
		# Improved_Estimate = ( r + self.gamma * later_rewards)
		improved_estimate = (r + (self.gamma * later_rewards))      # Correct
		result = Part_1 + (self.alpha * improved_estimate)
		if self.verbose:
			print "s: {}\na: {}\ns_prime: {}\nr: {}\nresult: {}".format(s, a, s_prime, r, result)
		self.q_table[s, a] = result
		return
	
	
def testLearner():
	qLearner = QLearner(num_states=100, num_actions=4, alpha=0.2, rar=0.98, radr=0.999, dyna=0, verbose=True)
	print qLearner.author()
	print qLearner.q_table
	print
	print qLearner.perform_random_action_or_not()
	print qLearner.update_q_table(s=6, a=2, s_prime=3, r=4)
	print qLearner.query(s_prime=3, r=5)
	print qLearner.querysetstate(3)
	
	
if __name__ == "__main__":
	testLearner()
	print "Remember Q from Star Trek? Well, this isn't him"
