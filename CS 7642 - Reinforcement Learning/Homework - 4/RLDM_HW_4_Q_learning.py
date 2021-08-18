#!/usr/bin/env python
# coding: utf-8

# ##### Reinforcement Learning and Decision Making &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Homework #4
# 
# # &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Q-Learning

# ## Description
# 
# In this homework, you will have the complete reinforcement-learning experience:  training an agent from scratch to solve a simple domain using Q-learning.
# 
# The environment you will be applying Q-learning to is called [Taxi](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py) (Taxi-v3).  The Taxi problem was introduced by [Dietterich 1998](https://www.jair.org/index.php/jair/article/download/10266/24463) and has been used for reinforcement-learning research in the past.  It is a grid-based environment where the goal of the agent is to pick up a passenger at one location and drop them off at another.
# 
# The map is fixed and the environment has deterministic transitions.  However, the distinct pickup and drop-off points are chosen randomly from 4 fixed locations in the grid, each assigned a different letter.  The starting location of the taxicab is also chosen randomly.
# 
# The agent has 6 actions: 4 for movement, 1 for pickup, and 1 for drop-off.  Attempting a pickup when there is no passenger at the location incurs a reward of -10.  Dropping off a passenger outside one of the four designated zones is prohibited, and attempting it also incurs a reward of −10.  Dropping the passenger off at the correct destination provides the agent with a reward of 20.  Otherwise, the agent incurs a reward of −1 per time step.
# 
# Your job is to train your agent until it converges to the optimal state-action value function.  You will have to think carefully about algorithm implementation, especially exploration parameters.
# 
# ## Q-learning
# 
# Q-learning is a fundamental reinforcement-learning algorithm that has been successfully used to solve a variety of  decision-making  problems.   Like  Sarsa,  it  is  a  model-free  method  based  on  temporal-difference  learning. However, unlike Sarsa, Q-learning is *off-policy*, which means the policy it learns about can be different than the policy it uses to generate its behavior.  In Q-learning, this *target* policy is the greedy policy with respect to the current value-function estimate.
# 
# ## Procedure
# 
# - You should return the optimal *Q-value* for a specific state-action pair of the Taxi environment.
# 
# - To solve this problem you should implement the Q-learning algorithm and use it to solve the Taxi environment. The agent  should  explore  the MDP, collect data  to  learn an optimal  policy and also the optimal Q-value function.  Be mindful of how you handle terminal states: if $S_t$ is a terminal state, then $V(St)$ should always be 0.  Use $\gamma= 0.90$ - this is important, as the optimal value function depends on the discount rate.  Also, note that an $\epsilon$-greedy strategy can find an optimal policy despite finding sub-optimal Q-values.   As we are looking for optimal  Q-values, you will have to carefully consider your exploration strategy.
# 
# ## Resources
# 
# The concepts explored in this homework are covered by:
# 
# -   Lesson 4: Convergence
# 
# -   Lesson 7: Exploring Exploration
# 
# -   Chapter 6 (6.5 Q-learning: Off-policy TD Control) of http://incompleteideas.net/book/the-book-2nd.html
# 
# -   Chapter 2 (2.6.1 Q-learning) of 'Algorithms for Sequential Decision Making', M.
#     Littman, 1996
# 
# ## Submission
# 
# -   The due date is indicated on the Canvas page for this assignment.
#     Make sure you have set your timezone in Canvas to ensure the deadline is accurate.
# 
# -   Submit your finished notebook on Gradescope. Your grade is based on
#     a set of hidden test cases. You will have unlimited submissions -
#     only the last score is kept.
# 
# -   Use the template below to implement your code. We have also provided
#     some test cases for you. If your code passes the given test cases,
#     it will run (though possibly not pass all the tests) on Gradescope. 
#     Be cognisant of performance.  If the autograder fails because of memory 
#     or runtime issues, you need to refactor your solution
# 
# -   Gradescope is using *python 3.6.x*, *gym==0.17.2* and *numpy==1.18.0*, and you can
#     use any core library (i.e., anything in the Python standard library).
#     No other library can be used.  Also, make sure the name of your
#     notebook matches the name of the provided notebook.  Gradescope times
#     out after 10 minutes.
# 

# In[21]:

################
# DO NOT REMOVE
# Versions
# gym==0.17.2
# numpy==1.18.0
################
import gym
import numpy as np


class QLearningAgent(object):
	def __init__(self):
		self.whoami = 'Jadams334'
		self.Q = None
		self.reset_chance = None
		self.use_random_restart = False
		self.decay_rate = None
		self.alpha = None
		self.alpha_decay = None
		self.original_alpha = None
		self.gamma = None
		self.epsilon = None
		self.epsilon_decay = None
		self.original_epsilon = None
		self.converge_thresh = None
		self.iteration = None
		self.num_episodes = None
		self.decay_to_use = None
		self.env = gym.make("Taxi-v3").env
	
	def get_action(self, state):
		if np.random.random() < self.epsilon:
			return np.random.randint(self.env.action_space.n)
		else:
			return np.argmax(self.Q[state, :])
	
	def solve(self):
		"""Create the Q table"""
		self.alpha = 0.2
		self.alpha_decay = 1e-2
		self.original_alpha = np.copy(self.alpha)
		self.epsilon = 0.9
		self.epsilon_decay = 1e-2
		self.original_epsilon = np.copy(self.epsilon)
		self.gamma = 0.9
		self.converge_thresh = 1e-15
		self.iteration = 0
		self.decay_to_use = 1
		self.decay_rate = 3
		self.reset_chance = 0.1
		self.use_random_restart = False
		
		# Initialize Q Table
		self.Q = np.zeros(shape=(self.env.observation_space.n, self.env.action_space.n))
		
		previous_n = 5
		previous_n_results = np.zeros(shape=(previous_n, 1))
		previous_n_count = 0
		converged = False
		converge_check_val = 0.2
		converge_display_results = []
		while not converged:
			# Initialize S
			state = self.env.reset()
			
			# Choose A from S using policy derived from Q (eps-greedy)
			action = self.get_action(state=state)
			episode_over = False
			prev_q = np.copy(self.Q)
			# Repeat for each step of episode
			self.num_episodes = 0
			while not episode_over:
				if self.use_random_restart:
					if np.random.random() < self.reset_chance:
						self.epsilon = np.copy(self.original_epsilon)
						self.alpha = np.copy(self.original_alpha)
				if self.epsilon <= 0:
					self.epsilon = np.copy(self.original_epsilon)
					self.epsilon_decay *= 0.9
				if self.alpha <= 0:
					self.alpha = np.copy(self.original_alpha)
					self.alpha_decay *= 0.9
				# Take Action A, observe R, S'
				next_state, reward, episode_over, misc = self.env.step(action)
				if self.decay_to_use == 0:
					if self.num_episodes % self.decay_rate == 0 and self.num_episodes > 0:
						self.alpha -= self.alpha_decay
						self.epsilon -= self.epsilon_decay
				
				elif self.decay_to_use == 1:
					if self.num_episodes % self.decay_rate == 0 and self.num_episodes > 0:
						self.alpha = np.copy(self.original_alpha)
						self.epsilon = np.copy(self.original_epsilon)
					self.alpha -= self.alpha_decay
					self.epsilon -= self.epsilon_decay
				
				elif self.decay_to_use == 2:
					if self.num_episodes % self.decay_rate == 0 and self.num_episodes > 0:
						self.alpha -= self.alpha_decay
				
				elif self.decay_to_use == 3:
					if self.num_episodes % self.decay_rate == 0 and self.num_episodes > 0:
						self.epsilon -= self.epsilon_decay
				
				elif self.decay_to_use == 4:
					pass
				
				# Update Q Values based on formula
				# target_value = reward + gamma * self.Q_table[next_state, next_action]
				self.Q[state, action] = self.Q[state, action] + self.alpha * (
						(reward + self.gamma * self.Q[next_state, np.argmax(self.Q[next_state, :])]) - self.Q[
					state, action])
				# self.Q[next_state, :] = 0
				state = next_state
				action = self.get_action(state=next_state)
				self.num_episodes += 1
			# Check Converged
			total_diff = np.mean(np.abs(self.Q - prev_q))
			previous_n_results[previous_n_count, 0] = total_diff
			if previous_n_count >= previous_n - 1:
				if previous_n_count == previous_n - 1:
					converge_check_val = np.mean(previous_n_results)
					previous_n_count = 0
				else:
					previous_n_count += 1
			else:
				previous_n_count += 1
			self.iteration += 1
			
			# if self.decay_to_use == 0:
			#     if self.iteration % 10 == 0 and self.iteration > 0:
			#         self.alpha -= self.alpha_decay
			#         self.epsilon -= self.epsilon_decay
			#
			# elif self.decay_to_use == 1:
			#     if self.iteration % 10 == 0 and self.iteration > 0:
			#         self.alpha = np.copy(self.original_alpha)
			#         self.epsilon = np.copy(self.original_epsilon)
			#     self.alpha -= self.alpha_decay
			#     self.epsilon -= self.epsilon_decay
			#
			# elif self.decay_to_use == 2:
			#     if self.iteration % 10 == 0 and self.iteration > 0:
			#         self.alpha -= self.alpha_decay
			#
			# elif self.decay_to_use == 3:
			#     if self.iteration % 10 == 0 and self.iteration > 0:
			#         self.epsilon -= self.epsilon_decay
			#
			# elif self.decay_to_use == 4:
			#     continue
			
			converge_display_results.append(converge_check_val)
			# if self.iteration % 1000 == 0 and self.iteration > 1:
			#     print(f"\tCurrent Iteration: {self.iteration}")
			
			if np.isclose(converge_check_val, 0, atol=self.converge_thresh):
				converged = True
	
	# print("Plotting Results")
	# self.plot_results(np.asarray(converge_display_results))
	
	def Q_table(self, state, action):
		"""return the optimal value for State-Action pair in the Q Table"""
		return self.Q[state][action]

# def plot_results(self, array):
#
# 	fig, ax = plt.subplots()
# 	ax.plot(np.round(np.arange(0, array.shape[0]), 1), array)
# 	ax.set_ylabel("Error")
# 	ax.set_xlabel("Iterations")
# 	decay_string = ""
# 	if self.decay_to_use == 0:
# 		decay_string = "Continuous Alpha and Epsilon Decay"
# 	elif self.decay_to_use == 1:
# 		decay_string = "Continuous Alpha and Epsilon Decay with Reset"
# 	elif self.decay_to_use == 2:
# 		decay_string = "Alpha Decay Only"
# 	elif self.decay_to_use == 3:
# 		decay_string = "Epsilon Decay Only"
# 	elif self.decay_to_use == 4:
# 		decay_string = "No Decay Used"
# 	ax.set_title(f"Q-Learner Taxi-v3\n{decay_string}")
# 	plt.savefig(f"Error_using_decay_schedule_{self.decay_to_use}.png")
# 	plt.show()


# ## 2. Test cases

# In[ ]:


## DO NOT MODIFY THIS CODE.  This code will ensure that you submission is correct
## and will work proberly with the autograder

import unittest


class TestQNotebook(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.agent = QLearningAgent()
		cls.agent.solve()
	
	def test_case_1(self):
		np.testing.assert_almost_equal(
			self.agent.Q_table(462, 4),
			-11.374402515,
			decimal=3
		)
	
	def test_case_2(self):
		np.testing.assert_almost_equal(
			self.agent.Q_table(398, 3),
			4.348907,
			decimal=3
		)
	
	def test_case_3(self):
		np.testing.assert_almost_equal(
			self.agent.Q_table(253, 0),
			-0.5856821173,
			decimal=3
		)
	
	def test_case_4(self):
		np.testing.assert_almost_equal(
			self.agent.Q_table(377, 1),
			9.683,
			decimal=3
		)
	
	def test_case_5(self):
		np.testing.assert_almost_equal(
			self.agent.Q_table(83, 5),
			-13.9968,
			#            -12.8232,
			decimal=3
		)


unittest.main(argv=[''], verbosity=2, exit=False)

# In[ ]:
