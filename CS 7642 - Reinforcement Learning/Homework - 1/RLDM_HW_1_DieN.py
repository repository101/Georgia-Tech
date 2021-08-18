#!/usr/bin/env python
# coding: utf-8

# #### Reinforcement Learning and Decision Making &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Homework #1
# 
# # &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Planning in MDPs

# ## Description
# 
# You are given an $N$-sided die, along with a corresponding Boolean mask
# vector, `is_bad_side` (i.e., a vector of ones and zeros). You can assume
# that $1<N\leq30$, and the vector `is_bad_side` is also of size $N$ and
# $1$ indexed (since there is no $0$ side on the die). The game of DieN is
# played as follows:
# 
# 1.  You start with $0$ dollars.
# 
# 2.  At any time you have the option to roll the die or to quit the game.
# 
#     1.  **ROLL**:
# 
#         1.  If you roll a number not in `is_bad_side`, you receive that
#             many dollars (e.g., if you roll the number $2$ and $2$ is
#             not a bad side -- meaning the second element of the vector
#             `is_bad_side` is $0$, then you receive $2$ dollars). Repeat
#             step 2.
# 
#         2.  If you roll a number in `is_bad_side`, then you lose all the
#             money obtained in previous rolls and the game ends.
# 
#     2.  **QUIT**:
# 
#         1.  You keep all the money gained from previous rolls and the
#             game ends.
# 
# ## Procedure
# 
# -   You will implement your solution using the `solve()` method
#     in the code below.
#     
# -   Your return value should be the number of dollars you expect to
#     win for a specific value of `is_bad_side`, if you follow an
#     optimal policy. That is, what is the value of the optimal
#     state-value function for the initial state of the game (starting
#     with $0$ dollars)? Your answer must be correct to $3$ decimal
#     places, truncated (e.g., 3.14159265 becomes 3.141).
# 
# -   To solve this problem, you will need to determine an optimal policy
#     for the game of DieN, given a particular configuration of the die.
#     As you will see, the action that is optimal will depend on your
#     current bankroll (i.e., how much money you've won so far).
# 
# -   You can try solving this problem by creating an MDP of the game
#     (states, actions, transition function, reward function, and assume a
#     discount rate of $\gamma=1$) and then calculating the optimal
#     state-value function.
# 
# ## Resources
# 
# The concepts explored in this homework are covered by:
# 
# -   Lecture Lesson 1: Smoov & Curly's Bogus Journey
# 
# -   Chapter 3 (3.6 Optimal Policies and Optimal Value Functions) and
#     Chapter 4 (4.3-4.4 Policy Iteration, Value Iteration) of
#     http://incompleteideas.net/book/the-book-2nd.html
# 
# -   Chapters 1-2 of 'Algorithms for Sequential Decision Making', M.
#     Littman, 1996
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
# 
# -   Gradescope is using *python 3.6.x* and *numpy==1.18.0*, and you can
#     use any core library (i.e., anything in the Python standard library).
#     No other library can be used.  Also, make sure the name of your
#     notebook matches the name of the provided notebook.  Gradescope times
#     out after 10 minutes.

# In[5]:


#################
# DO NOT REMOVE
# Versions
# numpy==1.18.0
################
import numpy as np
import sys
import os


class MDPAgent(object):
    def __init__(self):
        self.gamma = 1
        self.max_states = 101
        self.is_bad_side = None
        self.number_of_sides_on_dice = None
        self.chance_to_lose_all_rewards = None
        self.Value_Lookup = {}
        self.Value_Matrix = None
        self.Transition_Lookup = {}
        self.Transition_Matrix = None
        self.Reward_Lookup = {}
        self.Reward_Matrix = None
        self.Policy_Lookup = {}
        self.Policy_Matrix = None
        self.V = {}
        self.alt_optimal_policy = {}
        self.actions = {"Roll": 1, "Quit": 0}
        self.good_sides = None
        self.states = []
        self.conditions = {"Playable": 1, "Ended": 0}
        self.all_states = []
        self.states_visited_during_policy_evaluation = set()
        self.Memoization = {}
    
    def setup(self, is_bad_side):
        try:
            self.is_bad_side = np.asarray(is_bad_side)
            self.number_of_sides_on_dice = self.is_bad_side.shape[0]
            self.chance_to_lose_all_rewards = np.count_nonzero(self.is_bad_side == 1) / self.number_of_sides_on_dice
            self.good_sides = (np.argwhere(self.is_bad_side == 0) + 1)[:, 0].astype(np.int)
            self.Reward_Matrix = np.zeros(shape=(self.max_states, 2))
            self.Transition_Matrix = np.zeros(shape=(self.max_states, self.max_states))
            self.Policy_Matrix = np.full(shape=(self.max_states, 1), fill_value="Quit", dtype=object)
        except Exception as _exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'setup'", _exception)

    @staticmethod
    def create_state(value, condition):
        try:
            if condition == 1 or condition == "Playable":
                return value, "Playable"
            elif condition == 0 or condition == "Ended":
                return value, "Ended"
            else:
                return None
        except Exception as _exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'create_state'", _exception)

    def get_transition_probability(self, state, action, next_state):
        try:
            # The transition model which returns the probability of reaching next_state, given currently at state and
            #    took action.
            if state[1] == "Playable":
                if action == "Roll":
                    if state == next_state:
                        # Because you cannot roll a 0, thus no way to reach same state
                        return 0.0
                    elif next_state[1] == "Ended":
                        return self.chance_to_lose_all_rewards
                    elif next_state[1] == "Playable":
                        return 1 / self.number_of_sides_on_dice
                elif action == "Quit":
                    # The only valid result from quitting from state s would be the same bank with a 'Ended' condition
                    if next_state == (state[0], "Ended"):
                        return 1.0
                    else:
                        return 0.0
            elif state[1] == "Ended":
                return 0.0

        except Exception as _exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'get_transition_probability'", _exception)
  
    def get_next_states(self, state, action):
        try:
            test_val = np.where(self.is_bad_side == 0)[0]
            if test_val.shape[0] > 0 and state[1] == "Playable" and state[0] < self.max_states - np.max(test_val + 1) and action == "Roll" or action == self.actions["Roll"]:
                next_states = [self.create_state(s, "Playable") for s in self.good_sides + state[0]]
                return next_states
            else:
                return [self.create_state(state[0], "Ended")]
        except Exception as _exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'get_next_states'", _exception)

    def get_estimated_reward(self, state, action):
        try:
            # if state[1] == "Playable" and action == "Quit":
            #     return state[0]
            # else:
            #     return 0
            if state[1] == "Playable":
                if action == "Quit":
                    return 0
                else:
                    return (self.chance_to_lose_all_rewards * (state[0] * -1)) + np.sum(self.good_sides * (1 / self.number_of_sides_on_dice))
            else:
                return 0

        except Exception as _exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'get_estimated_reward'", _exception)

    def populate_transition_lookup(self, start_state):
        try:
            # Establishing the chance to lose for any reachable state
            visited = set(start_state)
            
            # adding start state
            queue = [start_state]
            keep_populating = True
            while keep_populating:
                if len(queue) > 0:
                    current_state = queue.pop(0)
                    self.all_states.append(current_state)
                    for action, _ in self.actions.items():
                        estimated_reward = self.get_estimated_reward(current_state, action)
                        self.Reward_Lookup[current_state, action] = estimated_reward
                        self.Reward_Matrix[current_state[0], self.actions[action]] = estimated_reward
                        lost_state = self.create_state(0, self.conditions["Ended"])
                        if current_state[1] != "Ended":
                            self.Transition_Matrix[current_state[0], lost_state[0]] = self.get_transition_probability(current_state, action, lost_state)
                        next_states = self.get_next_states(current_state, action)
                        if next_states is not None and next_states != 0:
                            for s_prime in next_states:
                                if s_prime[0] >= self.max_states and s_prime not in visited:
                                    continue
                                transition_probability = self.get_transition_probability(current_state, action, s_prime)
                                self.Transition_Lookup[current_state, action, s_prime] = transition_probability
                                if current_state[1] != "Ended":
                                    self.Transition_Matrix[current_state[0], s_prime[0]] = transition_probability
                                if s_prime not in visited:
                                    queue.append(s_prime)
                                    visited.add(s_prime)
                        
                                if len(queue) <= 0:
                                    keep_populating = False
                    if len(queue) <= 0:
                        keep_populating = False
                else:
                    keep_populating = False
            return
        except Exception as _exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'populate_transition_lookup'", _exception)

    def populate_policy(self):
        try:
            for i in range(self.max_states):
                for condition, _ in self.conditions.items():
                    if condition == "Ended" or condition == self.conditions["Ended"]:
                        continue
                    else:
                        if ((i, condition), "Roll") in self.Reward_Lookup and ((i, condition), "Quit") in self.Reward_Lookup:
                            temp_table = np.asarray([self.Reward_Lookup[(i, condition), "Roll"], self.Reward_Lookup[(i, condition), "Quit"]])
                            if np.all(np.isnan(temp_table)):
                                continue
                            else:
                                if np.all(temp_table == i):
                                    self.Policy_Lookup[(i, condition)] = "Quit"
                                else:
                                    if temp_table[0] != temp_table[1]:
                                        action_to_take = np.nanargmax(temp_table)
                                        if action_to_take == 0:
                                            self.Policy_Lookup[(i, condition)] = "Roll"
                                        elif action_to_take == 1:
                                            
                                            self.Policy_Lookup[(i, condition)] = "Quit"
                                    else:
                                        self.Policy_Lookup[(i, condition)] = "Quit"
            return
        except Exception as _exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'populate_policy'", _exception)
            
    def run_value_iteration(self, theta=0.00001):
        try:
            # Theta determines the accuracy of estimation
    
            # Initialize V(s) for all s in S+, arbitrarily except at terminal states
            for current_state in self.all_states:
                if current_state[1] == "Ended":
                    self.V[current_state] = 0
                else:
                    self.V[current_state] = np.random.random()
                    
            temp_results = np.zeros(shape=[self.max_states, 2])
            delta = 5
            num_iter = 0
            while delta > theta:
                num_iter += 1
                delta = 0
                for current_state in self.all_states:
                    little_v = self.V[current_state]
                    results = np.zeros(shape=(2,))

                    for action, val in self.actions.items():
                        next_states = self.get_next_states(current_state, action)
                        rewards = np.asarray([self.Reward_Lookup[s_prime, action] for s_prime in next_states])
                        v_rewards = np.asarray([self.V[s_prime] for s_prime in next_states])
                        probability = np.asarray([self.Transition_Lookup[current_state, action, s_prime] for s_prime in next_states])
                        results[val] = np.sum((rewards + (self.gamma * v_rewards)) * probability)
                    self.V[current_state] = np.nanmax(results)
                    temp_results[current_state[0], 0] = results[0]
                    temp_results[current_state[0], 1] = results[1]
                    # delta += max(delta, np.abs(little_v - V[state]))
                    delta = max(delta, np.abs(little_v - self.V[current_state]))
            return
        except Exception as _exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'run_value_iteration'", _exception)

    def run_policy_iteration(self, start_state):
        try:
            return
        except Exception as _exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'run_policy_iteration'", _exception)
    
    def test_value_iteration(self, theta=0.00001):
        # Theta determines the accuracy of estimation
        
        # Initialize V(s) for all s in S+, arbitrarily except at terminal states
        for current_state in self.all_states:
            if current_state[1] == "Ended":
                self.V[current_state] = 0
            else:
                self.V[current_state] = np.random.random()
    
        temp_results = np.zeros(shape=[self.max_states, 2])
        diff = 0
        
        while True:
            delta = 0
            for state in self.all_states:
                little_v = self.V[state]
                if state in self.Policy_Lookup:
                    temp_value = -np.inf
                    for action, _ in self.actions.items():
                        reward = self.Reward_Lookup[state, action]
                        estimated_value = reward + (self.gamma * self.V[state])
                        if estimated_value > temp_value:
                            temp_value = estimated_value
                    self.V[state] = temp_value
                    current_state = max(delta, np.abs(little_v - self.V[state]))
                    diff = np.abs(delta - current_state)
            if diff < theta:
                break
    
    def get_optimal_policy(self):
        try:
            temp_results = np.zeros(shape=(self.max_states, 2))
            for state in self.all_states:
                if state[1] == "Playable":
                    temp = state[0], (1 / self.number_of_sides_on_dice) * np.sum(
                        np.asarray([s[0] for s in self.get_next_states(state, "Roll")]))
                    temp_results[state[0], 0] = state[0]
                    temp_results[state[0], 1] = temp[1]
                    if temp[0] > temp[1]:
                        self.alt_optimal_policy[state] = "Quit"
                    elif temp[0] == temp[1]:
                        self.alt_optimal_policy[state] = "Quit"
                    else:
                        self.alt_optimal_policy[state] = "Roll"
            return
        except Exception as _exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'get_optimal_policy'", _exception)
    
    def evaluate_policy(self, state):
        try:
            if len(self.alt_optimal_policy) == 0:
                self.alt_optimal_policy = self.Policy_Lookup
            if state in self.Memoization:
                return self.Memoization[state]
            else:
                if self.alt_optimal_policy[state] == "Roll":
                    values = []
                    t = self.get_next_states(state, "Roll")
                    for s_prime in t:
                        if self.alt_optimal_policy[s_prime] == "Roll":
                            tmp = (1 / self.number_of_sides_on_dice * self.evaluate_policy(s_prime))
                        else:
                            tmp = (1 / self.number_of_sides_on_dice * s_prime[0])
                        values.append(tmp)
                    self.Memoization[state] = np.sum(np.asarray(values))
                    return self.Memoization[state]
                else:
                    return state[0]
        except Exception as _exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in 'evaluate_policy'", _exception)

    def solve(self, is_bad_side):
        """Implement the agent"""
        self.setup(is_bad_side=is_bad_side)
        start_state = self.create_state(0, "Playable")
        self.populate_transition_lookup(start_state)
        self.populate_policy()
        # self.test_value_iteration()
        self.run_value_iteration()
        self.get_optimal_policy()
        result = self.evaluate_policy(start_state)
        return result


# ## Test cases
# 
# We have provided some test cases for you to help verify your implementation.

# In[ ]:


## DO NOT MODIFY THIS CODE.  This code will ensure that your submission
## will work proberly with the autograder

import unittest

class TestDieNNotebook(unittest.TestCase):
    def test_case_0(self):
        agent = MDPAgent()
        np.testing.assert_almost_equal(
            agent.solve(is_bad_side=[0, 0, 0, 0, 1, 0, 0]),
            9.334,
            decimal=3
        )

    def test_case_1(self):
        agent = MDPAgent()
        np.testing.assert_almost_equal(
            agent.solve(is_bad_side=[1, 1, 1, 0, 0, 0]),
            2.583,
            decimal=3
        )

    def test_case_2(self):
        agent = MDPAgent()
        np.testing.assert_almost_equal(
            agent.solve(
                is_bad_side=[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
            ),
            7.379,
            decimal=3
        )

    def test_case_3(self):
        agent = MDPAgent()

        np.testing.assert_almost_equal(
            agent.solve(
                is_bad_side=[1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
            ),
            6.314,
            decimal=3
        )

    def test_case_4(self):
        agent = MDPAgent()

        np.testing.assert_almost_equal(
            agent.solve(
                is_bad_side=[0, 0, 0, 0, 1, 0, 0]
            ),
            9.334,
            decimal=3
        )

    def test_case_6(self):
        agent = MDPAgent()

        np.testing.assert_almost_equal(
            agent.solve(
                is_bad_side=[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0]
            ),
            5.965,
            decimal=3
        )

    def test_case_7(self):
        agent = MDPAgent()

        np.testing.assert_almost_equal(
            agent.solve(
                is_bad_side=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
            ),
           27.497,
            decimal=3
        )

    # def test_case_5(self):
    #     number_of_generations = 2000
    #     number_selected = 100
    #     visited = set()
    #     longest_time = 0
    #     array_for_longest_time = None
    #     for i in range(2, 30):
    #
    #         all_combinations = np.random.choice([0, 1], size=(number_of_generations, i))
    #         selected_idx = np.random.choice(np.arange(number_of_generations), size=number_selected, replace=False)
    #
    #         for idx in range(number_selected):
    #             c, v = np.unique(all_combinations[selected_idx[idx], :], return_counts=True)
    #             if np.any(c == 1):
    #                 sides_as_tuple = tuple(all_combinations[idx, :])
    #                 if sides_as_tuple not in visited:
    #                     agent = MDPAgent()
    #                     start_time = time.time()
    #                     print(f"Currently Processing: {all_combinations[selected_idx[idx], :]}")
    #                     result = agent.solve(is_bad_side=all_combinations[selected_idx[idx], :])
    #                     end_time = time.time()
    #                     elapsed_time = end_time - start_time
    #                     print(f"\tElapsed Time: {elapsed_time:.4f}s")
    #                     if elapsed_time > longest_time:
    #                         longest_time = elapsed_time
    #                         array_for_longest_time = all_combinations[selected_idx[idx], :]
    #                     visited.add(sides_as_tuple)
    #     longest_list = array_for_longest_time.tolist()

unittest.main(argv=[''], verbosity=2, exit=False)

