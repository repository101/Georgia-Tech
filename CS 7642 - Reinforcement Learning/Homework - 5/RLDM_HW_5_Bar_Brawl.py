#!/usr/bin/env python
# coding: utf-8

# ##### Reinforcement Learning and Decision Making &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Homework #5
# 
# # &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Bar Brawl

# ## Description
# 
# You are the proprietor of an establishment that sells beverages of an unspecified, but delicious, nature. The establishment is frequented by a set $P$ of patrons.  One of the patrons is the instigator and another is the peacemaker.
# 
# On any given evening, a subset $S \subseteq P$ is present at the establishment. If the instigator is in $S$ but the peacemaker is not in $S$, then a fight will break out. If the instigator is not in $S$ or if the peacemaker is in $S$, then no fight will occur.
# 
# Your goal is to learn to predict whether a fight will break out among the subset of
# patrons present on a given evening, without initially knowing the identity of
# the instigator or the peacemaker.
# 
# ## Procedure
# 
# Develop a KWIK learner for this problem (see Li, Littman, and Walsh 2008).  Your learner will be presented with $S$, the patrons at the establishment, and the outcome (fight or no fight) of that evening.
# Your learner will attempt to predict whether a fight will break out, or indicate that
# it doesn't know, and should be capable of learning from the true outcome for the evening.
# 
# For each problem, the following input will be given:
# 
# -   `at_establishment`: a Boolean two-dimensional array whose rows
#     represent distinct evenings and whose columns represent distinct patrons.
#     Each entry specifies if a particular patron is present
#     at the establishment on a particular evening:
#     a $1$ means present and a $0$ means absent.
#     
# -   `fight_occurred`: a Boolean vector whose entries are the 
#     outcomes (a $1$ means ``FIGHT`` and a $0$ means ``NO FIGHT``) for that
#     particular evening.
# 
# 
# Specifically:
# 
# -   For each episode (evening), you should present your learner with the next row of `at_establishment` and the corresponding row of `fight_occurred`.
# -   If your learner returns a $1$ (for ``FIGHT``) or a $0$ (for ``NO FIGHT``),
#     you may continue on to the next episode.
# -   If your learner returns a $2$ (for ``I DON’T KNOW``), then you should
#     present the pair (`at_establishment`, `fight_occurred`) to you learner
#     to learn from.
# 
# You will return a string of integers corresponding to the returned values of each episode.
# 
# The test case will be considered successful if no wrong answers are returned
# **and** the number of "I DON'T KNOW"s does not exceed the maximum 
# allowed by the autograder.
# 
# ## Resources
# 
# The concepts explored in this homework are covered by:
# 
# -   'Knows what it knows: A framework for self-aware learning', Li, Littman, Walsh 2008
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
# -   Gradescope is using *python 3.6.x* and *numpy==1.18.0*, and you can
#     use any core library (i.e., anything in the Python standard library).
#     No other library can be used.  Also, make sure the name of your
#     notebook matches the name of the provided notebook.  Gradescope times
#     out after 10 minutes.

# In[6]:


################
# DO NOT REMOVE
# Versions
# numpy==1.18.0
################
import numpy as np
import itertools


class Agent(object):
    def __init__(self):
        self.HypothesisSpace = None
        self.lHat = None
        self.responseDict = {"I DON'T KNOW": 2, "FIGHT": 1, "NO FIGHT": 0}
        self.altResponseDict = {3: "NO FIGHT", 2: "NO FIGHT", 1: "FIGHT", 0: "NO FIGHT"}
        pass
    
    def cleanUpPermutations(self, perm):
        finalPerm = []
        for i in range(len(perm)):
            temp = np.asarray(perm[i])
            val, counts = np.unique(temp, return_counts=True)
            if val.shape[0] != min(len(perm[0]), 3):
                continue
            else:
                # Find Index in Val for 1
                oneCount = None
                if len(counts[val == 1]) > 0:
                    oneCount = counts[val == 1][0]
                # Find Index in val for 2
                twoCount = None
                if len(counts[val == 2]) > 0:
                    twoCount = counts[val == 2][0]
                if oneCount is None:
                    continue
                elif twoCount is None:
                    continue
                else:
                    # Need to check both oneCount and twoCount
                    if oneCount > 1 or twoCount > 1:
                        continue
                    else:
                        finalPerm.append(perm[i])
        return finalPerm

    def generatePermutations(self, length):
        permutations = list(itertools.product([0, 1, 2], repeat=length))
        cleanedP = self.cleanUpPermutations(permutations)
        return cleanedP
    
    def generateHypothesisSpace(self, val):
        self.HypothesisSpace = np.asarray(self.generatePermutations(length=len(val)))
        return
    
    def computeLHat(self, inputX):
        results = np.sum(inputX * self.HypothesisSpace, axis=1)
        return results
    
    def determineResponse(self):
        val = [self.responseDict[self.altResponseDict[i]] for i in self.lHat]
        valSet = set(val)
        if len(valSet) > 1:
            return self.responseDict["I DON'T KNOW"]
        else:
            return self.responseDict[self.altResponseDict[valSet.pop()]]

    def learnFromExperience(self, inX, label):
        if label == 0:
            self.HypothesisSpace = np.copy(self.HypothesisSpace[self.lHat != 1])
        elif label == 1:
            self.HypothesisSpace = np.copy(self.HypothesisSpace[self.lHat == label])
    
    def solve(self, at_establishment, fight_occurred):
        resultingString = ""
        self.generateHypothesisSpace(at_establishment[0])
        for count, (inputX, trueLabel) in enumerate(zip(at_establishment, fight_occurred)):
            self.lHat = self.computeLHat(np.asarray(inputX))
            response = self.determineResponse()
            resultingString += str(response)
            if response == self.responseDict["I DON'T KNOW"]:
                # Learn Things
                if self.HypothesisSpace.shape[0] > 1:
                    self.learnFromExperience(inX=inputX, label=trueLabel)
        return resultingString


# In[ ]:



## DO NOT MODIFY THIS CODE.  This code will ensure that you submission is correct 
## and will work proberly with the autograder

import unittest


class TestBarBrawl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = Agent()

    def test_case_1(self):
        np.testing.assert_equal(
            self.agent.solve(
                [[1,1], [1,0], [0,1], [1,1], [0,0], [1,0], [1,1]],
                [0, 1, 0, 0, 0, 1, 0]
            ),
            '0200010'
        )

    def test_case_2(self):
        np.testing.assert_equal(
             self.agent.solve(
                [[1,0,0,0,],[0,1,0,0],[0,1,1,1],[0,1,1,1],[0,1,1,1],[0,0,0,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[0,1,1,1],[1,1,1,1],[0,1,1,1],[0,1,1,1],[1,1,1,1],[0,1,1,1],[0,1,1,1],[1,1,1,1],[0,1,1,1],[0,1,1,1],[0,0,0,1],[0,0,0,1],[1,1,1,1],[0,1,1,1],[0,1,1,1],[0,0,1,1],[0,1,1,1],[0,0,0,1],[0,0,1,1],[0,1,1,1],[0,0,1,1],[1,1,1,1]],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0]
            ),
            '22200200000000000000000002001010'
        )

    def test_case_3(self):
        np.testing.assert_equal(
             self.agent.solve(
                [[1,0,1],[1,0,1],[1,1,1],[1,1,1],[1,1,1],[0,1,1],[0,0,1],[0,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[0,1,1],[1,0,1],[1,1,1],[1,0,1],[1,0,1],[1,1,1]],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            ),
            '200002000000000000'
        )
        
unittest.main(argv=[''], verbosity=2, exit=False)

