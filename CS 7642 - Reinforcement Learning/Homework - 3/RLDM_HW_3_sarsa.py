#!/usr/bin/env python
# coding: utf-8

# ##### Reinforcement Learning and Decision Making &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Homework #3
# 
# # &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sarsa

# ## Description
# 
# For this assignment,  you will build a Sarsa agent which will learn policies in the [OpenAI Gym](http://gym.openai.com/docs/) Frozen Lake environment.  [OpenAI Gym](http://gym.openai.com/docs/) is a platform where users can test their RL algorithms on a selection of carefully crafted environments.  As we will continue to use [OpenAI Gym](http://gym.openai.com/docs/) through Project 2, this assignment also provides an opportunity to familiarize yourself with its interface.
# 
# Frozen Lake is a grid world environment that is highly stochastic,  where the agent must cross a slippery frozen  lake  which  has  deadly  holes  to  fall  through.   The  agent  begins  in  the  starting  state `S` and  is  given  a reward of `1` if it reaches the goal state `G`.  A reward of `0` is given for all other transitions.
# 
# The agent can take one of four possible moves at each state (left, down, right, or up).  The frozen cells `F` are slippery, so the agent’s actions succeed only `1/3` of the time, while the other `2/3` are split evenly between the two directions orthogonal to the intended direction.  If the agent lands in a hole `H`, then the episode terminates. You will be given a randomized Frozen Lake map with a corresponding set of parameters to train your Sarsa agent with.  If your agent is implemented correctly, then after training it for the specified number of episodes, your agent will produce the same policy (not necessarily an optimal policy) as the automatic grader.
# 
# 
# ## Sarsa $($$S_t$, $A_t$, $R_{t+1}$, $S_{t+1}$, $A_{t+1}$$)$
# 
# Sarsa uses temporal-difference learning to form a model-free on-policy reinforcement-learning algorithm that solves the *control* problem. It is model free because it does not need and does not use a model of the environment, namely neither a transition nor reward function; instead, Sarsa samples transitions and rewards online.
# 
# It is on-policy because it learns about the same policy that generates its behaviors (this is in contrast to *Q-learning*, which you’ll examine in your next homework).  That is, Sarsa estimates the action-value function of its behavior policy.  In this homework,  you will not be training a Sarsa agent to approximate the *optimal* action-value function; instead, the hyperparameters of both the exploration strategy and the algorithm will be given to you as input — the goal being to verify that your SARSA agent is correctly implemented.
# 
# ## Procedure
# 
# Since this homework requires you to match a non-deterministic output between your agent and the autograder’s agent, attention to detail to each of the following points is required:
# 
# - You must use Python and the library NumPy for this homework *python 3.6.x* and *numpy==1.18.0*
# 
# - Install OpenAI Gym (e.g.pip install gym) *gym==0.17.2*
# 
# - The Frozen Lake environment has been instantiated for you.
# - The pertinent random number generators have been seeded for you. Do *not* use the Python standard library’s *random* library.
# 
# - Implement your Sarsa agent using an $\epsilon$-greedy behavioral policy.  Specifically, you must use *numpy.random.random* to  choose  whether  or  not  the  action  is  greedy,  and *numpy.random.randint* to select the random action.
# 
# - Initialize the agent’s Q-table to zeros.
# 
# - Train your agent using the given input parameters.  The input *amap* is the Frozen Lake map that you need to resize and provide to the *desc* attribute when you instantiate your environment.  The input *gamma* is the discount rate.  The input *alpha* is the learning rate.  The input *epsilon* is the parameter for the $\epsilon$-greedy behavior strategy your Sarsa agent will use.  Specifically, an action should be selected uniformly at random if a random number drawn uniformly between 0 and 1 is less than $\epsilon$.  If the greedy action is selected,  the  action  with  lowest  index  should  be  selected  in  case  of  ties.   The  input `n_episodes` is  the number of episodes to train your agent.  Finally, *seed* is the number used to seed both Gym’s random number generator and NumPy’s random number generator.
# 
# - To sync with the autograder,  your Sarsa implementation should select the action corresponding to the next state the  agent  will  visit *even when* that  next  state  is  a  terminal  state  (this  action  will  never  be executed by the agent).
# 
# - You should return the greedy policy with respect to the Q-function obtained by your Sarsa agent after the completion of the final episode.  Specifically, the policy should be expressed as a string ofcharacters: **<, v, >, ^,** representing left, down, right, and up, respectively.  The ordering of the actions in the output should reflect the ordering of states in *amap*. 
# 
# ## Resources
# 
# The concepts explored in this homework are covered by:
# 
# -   Lesson 4: Convergence
# 
# -   Chapter 6 (6.4 Sarsa:  On-policy TD Control) of
#     http://incompleteideas.net/book/the-book-2nd.html
# 
# 
# ## Submission
# 
# -   The due date is indicated on the Canvas page for this assignment.
#     Make sure you have your timezone in Canvas set to ensure the
#     deadline is accurate.
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

# In[1]:


#################
# DO NOT REMOVE
# Versions
# numpy==1.18.0
# gym==0.17.2
################
import random

import gym
import numpy as np
from math import sqrt
from gym.envs import toy_text


"""
NOTES
    Use np.random.random for epsilon-greedy choice
    Use np.random.randint for selecting the random action to take
    Initialize Q-table to all zeros, np.zeros()
    In case of ties, use action with lowest index
    The output policy should have an action even in terminal state
"""

"""
Mathematical Symbol Alt-Codes
    https://www.webnots.com/alt-code-shortcuts-for-mathematics-symbols/
    
PSEUDOCODE
    α - Alpha (alt+224)
    ε - Epsilon (alt+238)
    γ - Gamma (IDK how to make, just copy paste)
    Step size α (0,1], ε > 0
    Initialize Q(S,A) for all S in S', A in A(S), arbitrarily except that Q(terminal_state) = 0
    For each episode do:
        Initialize S
        Choose A from S using policy derived from Q
        For each step of episode do:
            Take action A, observe R, S'
            Chose A' from S' using policy derived from Q
            Q(S,A) = Q(S, A) + α[R + γQ(S',A') - Q(S,A)]
            S = S'
            A = A'
            Until S is Terminal
        end
    end

"""


class FrozenLakeAgent(object):
    def __init__(self):
        self.whoami = 'jadams334'
        self.Q_table = None
        self.action_to_character = {0: "<",
                                    1: "v",
                                    2: ">",
                                    3: "^"}
        self.epsilon = None
        self.gamma = None
        self.alpha = None
        self.env = None

    def amap_to_gym(self, amap='FFGG'):
        """Maps the `amap` string to a gym env"""
        amap = np.asarray(amap, dtype='c')
        side = int(sqrt(amap.shape[0]))
        amap = amap.reshape((side, side))
        return gym.make('FrozenLake-v0', desc=amap).unwrapped

    def get_final_policy(self):
        final_string = ""
        for i in range(len(self.Q_table)):
            final_string += self.action_to_character[np.argmax(self.Q_table[i, :])]
        return final_string
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.action_space.n)
        else:
            return np.argmax(self.Q_table[state, :])
        
    def solve(self, amap, gamma, alpha, epsilon, n_episodes, seed):
        """Implement the agent"""
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.env = self.amap_to_gym(amap)
        
        # Initialize Q Table
        self.Q_table = np.zeros(shape=(self.env.observation_space.n, self.env.action_space.n))
        np.random.seed(seed)
        self.env.seed(seed)
        for _ in range(n_episodes):
            # Initialize S
            state = self.env.reset()
            
            # Choose A from S using policy derived from Q (eps-greedy)
            action = self.get_action(state=state)
            episode_over = False
            
            # Repeat for each step of episode
            while not episode_over:
                # Take Action A, observe R, S'
                next_state, reward, episode_over, _ = self.env.step(action)
                # Choose A' from S', using policy derived from Q
                next_action = self.get_action(state=next_state)

                # Update Q Values based on formula
                # target_value = reward + gamma * self.Q_table[next_state, next_action]
                self.Q_table[state, action] = self.Q_table[state, action] + self.alpha * ((reward + self.gamma * self.Q_table[next_state, next_action]) - self.Q_table[state, action])
                state = next_state
                action = next_action
            
        # Helper function to convert policy to string
        policy = self.get_final_policy()
        return policy


# ## Test cases

# In[ ]:



## DO NOT MODIFY THIS CODE.  This code will ensure that you submission is correct
## and will work proberly with the autograder

import unittest


class TestQNotebook(unittest.TestCase):
    def setUp(self):
        self.agent = FrozenLakeAgent()

    def test_case_1(self):
        example1 = self.agent.solve(
            amap='SFFFHFFFFFFFFFFG',
            gamma=1.0,
            alpha=0.25,
            epsilon=0.29,
            n_episodes=14697,
            seed=741684
        )
        assert(example1 == '^vv><>>vvv>v>>><')

    def test_case_2(self):
        example2 = self.agent.solve(
            amap='SFFFFHFFFFFFFFFFFFFFFFFFG',
            gamma=0.91,
            alpha=0.12,
            epsilon=0.13,
            n_episodes=42271,
            seed=983459
        )
        assert(example2 == '^>>>><>>>vvv>>vv>>>>v>>^<')

    def test_case_3(self):
        example3 = self.agent.solve(
            amap='SFFG',
            gamma=1.0,
            alpha=0.24,
            epsilon=0.09,
            n_episodes=49553,
            seed=20240
        )
        assert(example3 == '<<v<')

    def test_case_4(self):
        example4 = self.agent.solve(
            amap='SFFHHFFHHFFHHFFG',
            gamma=0.99,
            alpha=0.5,
            epsilon=0.29,
            n_episodes=23111,
            seed=44323
        )
        assert(example4=='^><<<>^<<><<<>^<')

    def test_case_5(self):
        example5 = self.agent.solve(
            amap='SFFFFHFFFHHFFFFFFFFHHFFFG',
            gamma=0.88,
            alpha=0.15,
            epsilon=0.16,
            n_episodes=112312,
            seed=6854343
        )
        assert(example5 == '^>><^<>><<<>v<^v>v<<<>vv<')

unittest.main(argv=[''], verbosity=2, exit=False)



# In[ ]:




