import collections
import time


class SemanticNetsAgent:
	def __init__(self):
		self.initial_sheep = 0
		self.initial_wolves = 0
		self.visited_states = set()
		self.state_queue = None
		self.is_solved = False
		self.solved_state = None
		pass

	def generate_transitions_using_matrix(self, matrix):
		states = []
		all_moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]
		for current_move in all_moves:
			if matrix[4] == "Left Side":
				# Moving from left side to right side
				transition = ((matrix[0][0] - current_move[0], matrix[0][1] + current_move[0]),
				              (matrix[1][0] - current_move[1], matrix[1][1] + current_move[1]), matrix, current_move,
				              "Right Side")
				# moves.append(transition)
				if self.validate_generated_transition(transition):
					states.append(transition)
			elif matrix[4] == "Right Side":
				transition = ((matrix[0][0] + current_move[0], matrix[0][1] - current_move[0]),
				              (matrix[1][0] + current_move[1], matrix[1][1] - current_move[1]),
				              matrix, current_move, "Left Side")
				# moves.append(transition)
				if self.validate_generated_transition(transition):
					states.append(transition)
		return states

	def validate_generated_transition(self, transition):
		if (transition[0], transition[1], transition[4]) in self.visited_states or \
				transition[0][0] < 0 or transition[0][0] > self.initial_sheep or transition[0][1] < 0 or \
				transition[0][1] > self.initial_sheep or \
				transition[1][0] < 0 or transition[1][0] > self.initial_wolves or \
				transition[1][1] < 0 or transition[1][1] > self.initial_wolves or (
				0 < transition[0][0] < transition[1][0]) or \
				(0 < transition[0][1] < transition[1][1]) or \
				(transition[4] == "Left Side" and (transition[0][0] + transition[1][0]) == 1) or \
				(transition[4] == "Right Side" and (transition[0][1] + transition[1][1]) == 1):
			return False
		else:
			return True

	def get_path(self, state, path):
		current_state = state
		while current_state[2] != "Root":
			path.append(current_state[3])
			current_state = current_state[2]

		return path
		# if state[2] == "Root":
		# 	return path
		# else:
		# 	path.append(state[3])
		# 	return self.get_path(state[2], path)

	"""
		Pseudocode
		
		def Agent_BFS(self, root):
			    Initialize STATE_QUEUE with the root
			    Initialize VISITED_STATES
			    Initialize flag SOLVED to False
			    While STATE_QUEUE not empty and not SOLVED:
			        GENERATE_STATES from STATE_QUEUE.POP
			        Iterate over generated states:
			            If state is solution:
			                SOLVED is TRUE
			            Add state to VISITED_STATES
			            Add state to STATE_QUEUE
			            
		def GENERATE_STATES(some_state):
				Initialize MOVES (1, 0), (2, 0), (1, 1), (0, 1), (0, 2)
				Initialize STATE_CONTAINER
				Iterate over MOVES:
					Create new state base on move
					Validate State
					if generated_state valid:
						Add generated_state to STATE_CONTAINER
				return STATE_CONTAINER
	"""

	def bfs(self, root):
		self.state_queue = collections.deque([root])
		while self.state_queue and not self.is_solved:
			if len(self.visited_states) > 0 and (self.state_queue[0][0], self.state_queue[0][1], self.state_queue[0][4]) in self.visited_states:
				pass
			else:
				self.visited_states.add((self.state_queue[0][0], self.state_queue[0][1], self.state_queue[0][4]))
			all_states = self.generate_transitions_using_matrix(self.state_queue.popleft())
			if len(all_states) > 0:
				for temp_state in all_states:
					if temp_state[0][1] == self.initial_sheep and temp_state[1][1] == self.initial_wolves:
						self.is_solved = True
						self.solved_state = temp_state
					self.state_queue.append(temp_state)
					self.visited_states.add((temp_state[0], temp_state[1], temp_state[4]))
		if self.is_solved is False:
			return []
		else:
			final_path = self.get_path(self.solved_state, [])
			return final_path

	def solve(self, initial_sheep, initial_wolves):
		# Add your code here! Your solve method should receive
		# the initial number of sheep and wolves as integers,
		# and return a list of 2-tuples that represent the moves
		# required to get all sheep and wolves from the left
		# side of the river to the right.
		#
		# If it is impossible to move the animals over according
		# to the rules of the problem, return an empty list of
		# moves.

		if initial_wolves > initial_sheep:
			# Because invalid state
			return []
		self.initial_sheep = initial_sheep
		self.initial_wolves = initial_wolves
		self.visited_states = set()
		self.state_queue = None
		self.is_solved = False
		self.solved_state = None

		# (Row of Sheep, Row of Wolves, Pointer to Parent, Move used to get to location, Boat Location)
		start_state_matrix = ((initial_sheep, 0), (initial_wolves, 0), "Root", None, "Left Side")
		# start_time = time.time()
		result = self.bfs(root=start_state_matrix)
		# end_time = time.time()
		# elapsed_time = end_time - start_time
		# print(f"Elapsed time for ({self.initial_sheep},{self.initial_wolves}): {elapsed_time:.5f}s")
		# print(result)
		return result