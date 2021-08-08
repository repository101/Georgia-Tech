import numpy as np
import os
import sys


def sub_evaluate_policy():
	return


def evaluate_policy(state, number_of_sides_on_dice, policy, roll_array, ev_array, memo):
	if policy[state, 0] == "Quit":
		return state
	else:
		next_states = roll_array + state
		if state in memoization:
			return memo[state]
		else:
			memo[state] = (1 / number_of_sides_on_dice) * np.sum(np.asarray([evaluate_policy(s_prime, number_of_sides_on_dice, policy, roll_array, ev_array, memo) for s_prime in next_states]))
			return (1 / number_of_sides_on_dice) * state
		
		
if __name__ == "__main__":
	is_bad_side = np.asarray([0,0,0,0,1,0,0])
	num_sides = is_bad_side.shape[0]
	rolls_array = np.where(is_bad_side != 1)[0] + 1
	num_good_sides = rolls_array.shape[0]
	probability_array = np.asarray([1/num_sides for _ in range(num_good_sides)])
	EV_Array = np.zeros(shape=(100, 1))
	Policy_Array = np.zeros(shape=(100, 1), dtype=object)
	memoization = {}
	for i in range(100):
		EV_Array[i, 0] = np.sum(probability_array * (rolls_array + i))
		if EV_Array[i, 0] > i:
			Policy_Array[i, 0] = "Roll"
		else:
			Policy_Array[i, 0] = "Quit"
	result = evaluate_policy(0, number_of_sides_on_dice=num_sides, policy=Policy_Array, roll_array=rolls_array, ev_array=EV_Array, memo=memoization)
	print()
	
	