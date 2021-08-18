#!/usr/bin/env python
# coding: utf-8

# ##### Reinforcement Learning and Decision Making &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Homework #6
# 
# # &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Rock Paper Scissors

# ## Description
# 
# Rock, Paper, Scissors is a popular game among kids.  It is also a good game to study Game Theory, Nash Equilibria, Mixed Strategies, and Linear Programming.
# 
# <img src=https://d1b10bmlvqabco.cloudfront.net/paste/jzfsa4a37jf4aq/b05d4e6a42d0dc2bc2bce1f2a0097a19cbf364038b8df44793e23d0552f55a52/rps.png width="400"/>
# 
# ## Procedure
# 
# For this assignment, you are asked to compute the Nash equilibrium for the given zero sum games.  You will be given the reward matrix 'R' for Player A.  Since this is a
# zero-sum game, Player B’s reward matrix will be the opposite (additive inverse) of Player A’s matrix.
# The first column of
# the matrix specifies player A's reward for playing rock against player B's rock (row 1),
# paper (row 2) and scissors (row 3).  The second column specifies player A's reward
# for playing paper, and the third column player A's reward for playing scissors.
# 
# - You need to find the ideal mixed strategy for the game. While there are different ways to calculate this, we will use Linear Programming in the hopes of preparing you for your final project. Use the Linear Programming solver CVXPY (<https://www.cvxpy.org/index.html>) to create a program that can solve Rock, Paper, Scissors games with arbitrary reward matrices.  For an example of how to create a linear program to solve Rock, Paper, Scissors, see Littman 1994.
# 
# - You will return a vector of the Nash equilibrium probabilities for player A found by your linear program.
# 
# - Your answer must be correct to $3$ decimal places, truncated (e.g., 3.14159265 becomes 3.141).
# 
# ## Resources
# 
# The concepts explored in this homework are covered by:
# 
# -  Lesson 11A: Game Theory
# 
# -  Lesson 11B: Game Theory Reloaded
# 
# -  'Markov games as a framework for multi-agent reinforcement learning', Littman 1994
# 
# -  'A polynomial-time Nash equilibrium algorithm for repeated games', Littman, Stone 2005
# 
# -  <https://www.cvxpy.org/short_course/index.html>
# 
# ## Submission
# 
# -   The due date is indicated on the Canvas page for this assignment.
#     Make sure you have your timezone in Canvas set to ensure the
#     deadline is accurate.
# 
# -   Submit your finished notebook on Gradescope. Your grade is based on
#     a set of hidden test cases. You will have unlimited submissions.
#     By default, the last score is kept.  You can also set a particular
#     submission as active in the submission history, in which case that
#     submission will determine your grade.
# 
# -   Use the template below to implement your code. We have also provided
#     some test cases for you. If your code passes the given test cases,
#     it will run (though possibly not pass all the tests) on Gradescope.
# 
# -   Gradescope is using *python 3.6.x*, *cvxpy==1.0.26*, and *numpy==1.18.0*, and you can
#     use any core library (i.e., anything in the Python standard library).
#     No other library can be used.  Also, make sure the name of your
#     notebook matches the name of the provided notebook.  Gradescope times
#     out after 10 minutes.

# In[12]:


################
# DO NOT REMOVE
# Versions
# cvxpy==1.0.26
# numpy==1.18.0
################
import numpy as np
import cvxpy as cp


class RPSAgent(object):
    def __init__(self):
        pass

    def solve(self, R):
        # Generate a random non-trivial linear program.
        n = 1
        A = np.vstack((np.asarray(R), np.asarray([[1, 1, 1],
                                                  [-1, -1, -1]])))
        A = np.hstack((np.asarray([[1, 1, 1, 0, 0]]).T, A))
        b = np.asarray([[0, 0, 0, 1, -1]]).T
        c = np.asarray([[1, 0, 0, 0]]).T

        # Define and solve the CVXPY problem.
        x = cp.Variable((4, 1))
        prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x <= b, x >= 0])
        prob.solve()
    
        return np.round(x.value[1:, 0], 3).tolist()
        

# In[ ]:


## DO NOT MODIFY THIS CODE.  This code will ensure that you submission is correct 
## and will work proberly with the autograder

import unittest

class TestRPS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = RPSAgent()

    def test_case_1(self):
        R = [
            [0,1,-1],[-1,0,1],[1,-1,0]
        ]

        np.testing.assert_almost_equal(
            self.agent.solve(R),
            np.array([0.333, 0.333, 0.333]),
            decimal=3
        )
        
    def test_case_2(self):
        R = [[0,  2, -1],
            [-2,  0,  1],
            [1, -1,  0]]
    
        np.testing.assert_almost_equal(
            self.agent.solve(R),
            np.array([0.250, 0.250, 0.500]),
            decimal=3
        )

unittest.main(argv=[''], verbosity=2, exit=False)

