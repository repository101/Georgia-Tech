import collections
import numpy as np


class SemanticNetsAgent:
	def __init__(self):
		self.initial_sheep = 0
		self.initial_wolves = 0
		self.connected_states = []
		self.visited_states = []
		self.state_queue = None
		self.path = []
		self.current_level = []
		self.is_solved = False
		self.solved_state = None
		self.one_sheep_from_left = np.asarray([[-1, 1], [0, 0]])
		self.two_sheep_from_left = np.asarray([[-2, 2], [0, 0]])
		self.one_wolf_from_left = np.asarray([[0, 0], [-1, 1]])
		self.two_wolves_from_left = np.asarray([[0, 0], [-2, 2]])
		self.one_each_from_left = np.asarray([[-1, 1], [-1, 1]])
		pass

	def generate_transitions_using_matrix(self, matrix):
		moves = []
		all_moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]
		for current_move in all_moves:
			if matrix[4] == "Left Side":
				# Moving from left side to right side
				if matrix[0][0] - current_move[0] < 0 or matrix[1][0] - current_move[1] < 0 or \
						matrix[0][1] + current_move[0] < 0 or matrix[1][1] + current_move[1] < 0 or\
						matrix[0][1] + current_move[0] > self.initial_sheep or \
						matrix[1][1] + current_move[1] > self.initial_wolves or \
						(matrix[0][1] + current_move[0] + matrix[1][1] + current_move[1]) == 1 or \
						(matrix[0][0] - current_move[0] < matrix[1][0] - current_move[1]) or\
						(matrix[0][1] + current_move[0] < matrix[1][1] + current_move[1]) or \
						((matrix[0][0] - current_move[0], matrix[0][1] + current_move[0]),
						 (matrix[1][0] - current_move[1], matrix[1][1] + current_move[1]), "Right Side")\
						in self.visited_states:
					pass
				else:
					transition = ((matrix[0][0] - current_move[0], matrix[0][1] + current_move[0]),
					              (matrix[1][0] - current_move[1], matrix[1][1] + current_move[1]), matrix, current_move,
					              "Right Side")
					# moves.append(transition)
					if (transition[0], transition[1], transition[4]) not in self.visited_states:
						moves.append(transition)
			elif matrix[4] == "Right Side":
				if matrix[0][0] + current_move[0] < 0 or matrix[1][0] + current_move[1] < 0 or \
						matrix[0][1] - current_move[0] < 0 or matrix[1][1] - current_move[1] < 0 or\
						matrix[0][1] - current_move[0] > self.initial_sheep or \
						matrix[1][1] - current_move[1] > self.initial_wolves or \
						(matrix[0][0] + current_move[0] + matrix[1][0] + current_move[1]) == 1 or \
						(matrix[0][0] + current_move[0] < matrix[1][0] + current_move[1]) or\
						(matrix[0][1] - current_move[0] < matrix[1][1] - current_move[1]) or \
						((matrix[0][0] + current_move[0], matrix[0][1] - current_move[0]),
						 (matrix[1][0] + current_move[1], matrix[1][1] - current_move[1]), "Left Side") \
						in self.visited_states:
					pass
				else:
					transition = ((matrix[0][0] + current_move[0], matrix[0][1] - current_move[0]),
					              (matrix[1][0] + current_move[1], matrix[1][1] - current_move[1]),
					              matrix, current_move, "Left Side")
					# moves.append(transition)
					if (transition[0], transition[1], transition[4]) not in self.visited_states:
						moves.append(transition)
		return moves

	def move_is_valid_matrix(self, matrix):
		# Move is not valid if any side results in negative values
		if matrix[0][0] < 0 or matrix[0][0] > self.initial_sheep or \
				matrix[0][1] < 0 or matrix[0][1] > self.initial_sheep or \
				matrix[1][0] < 0 or matrix[1][0] > self.initial_wolves or \
				matrix[1][1] < 0 or matrix[1][1] > self.initial_wolves:
			return False
		return True

	def validate_generated_moves_matrix(self, moves):
		non_valid_moves = []
		valid_moves = []
		for idx in range(len(moves)):
			# Check if Wolves out number sheep
			current_move = moves[idx]
			if self.validate_generated_transition(moves[idx]):
				valid_moves.append(moves[idx])
			else:
				non_valid_moves.append(moves[idx])
		return valid_moves

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
		if state[2] == "Root":
			return path
		else:
			path.append(state[3])
			return self.get_path(state[2], path)

	def bfs(self, root):
		self.state_queue = collections.deque([root])
		while self.state_queue and not self.is_solved:
			self.connected_states.append(self.state_queue[0])
			if len(self.visited_states) > 0 and (
					self.state_queue[0][0], self.state_queue[0][1], self.state_queue[0][4]) in self.visited_states:
				pass
			else:
				self.visited_states.append((self.state_queue[0][0], self.state_queue[0][3]))
			if self.state_queue[0][3] == -1:
				a = self.state_queue[0][0] + self.one_sheep_from_left
				b = self.state_queue[0][0] + self.two_sheep_from_left
				c = self.state_queue[0][0] + self.one_each_from_left
				d = self.state_queue[0][0] + self.one_wolf_from_left
				e = self.state_queue[0][0] + self.two_wolves_from_left
			else:
				a = self.state_queue[0][0] + (self.one_sheep_from_left * -1)
				b = self.state_queue[0][0] + (self.two_sheep_from_left * -1)
				c = self.state_queue[0][0] + (self.one_each_from_left * -1)
				d = self.state_queue[0][0] + (self.one_wolf_from_left * -1)
				e = self.state_queue[0][0] + (self.two_wolves_from_left * -1)
			for i in [a,b,c,d,e]:
				if np.any(self.visited_states == (i, self.state_queue[0][3] * -1)) or np.any(i < 0):
					pass
				else:
					self.state_queue.append((i, self.state_queue[0], None, (0, 1), self.state_queue[0][3] * -1))

			print()

			# all_states = self.generate_transitions_using_matrix(self.state_queue.popleft())
			# if len(all_states) > 0:
			# 	for temp_state in all_states:
			# 		if temp_state[0][1] == self.initial_sheep and temp_state[1][1] == self.initial_wolves:
			# 			self.is_solved = True
			# 			self.solved_state = temp_state
			# 		self.state_queue.append(temp_state)
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
		self.connected_states = []
		self.visited_states = []
		self.state_queue = None
		self.path = []
		self.current_level = []
		self.is_solved = False
		self.is_solved = False
		self.solved_state = None
		# Works but is very slow "(O_o)"

		# (Row of Sheep, Row of Wolves, Pointer to Parent, Move used to get to location, Boat Location)
		start_state = np.asarray([[initial_sheep, 0], [initial_wolves, 0]])
		start_state_matrix = (start_state, "Root", None, -1)
		return self.bfs(root=start_state_matrix)

