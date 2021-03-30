from queue import PriorityQueue


class State:
	def __init__(self, current_state, goal_state, goal_locations_by_name, goal_locations_by_index,
	             non_moveable_blocks=None, is_initial_state=False, previously_visited=False, number_of_moves=0):
		self.current_state = current_state
		self.is_initial_state = is_initial_state
		self.goal_state = goal_state
		self.number_of_moves = number_of_moves
		self.previously_visited = previously_visited
		self.goal_locations_by_name = goal_locations_by_name
		self.goal_locations_by_index = goal_locations_by_index
		self.is_goal_state = self.determine_if_goal_state()
		self.block_specific_error = {}
		self.current_locations_by_name = {}
		self.current_locations_by_index = {}
		if non_moveable_blocks is None:
			non_moveable_blocks = self.find_non_moveable()
		self.non_moveable_blocks = non_moveable_blocks
		self.delta = self.calculate_delta() + self.number_of_moves
		self.moveable_blocks = None
		self.get_current_state_locations()
		self.key = tuple((key, val) for key, val in self.current_locations_by_name.items())
		self.moves_used_to_get_to_state = []

	def __eq__(self, other):
		return self.delta == other.delta

	def __lt__(self, other):
		return self.delta < other.delta

	def __gt__(self, other):
		return self.delta > other.delta

	def __hash__(self):
		return hash(self.key)

	def find_non_moveable(self):
		if self.is_goal_state:
			return set()
		result = set()
		for i in range(len(self.current_state)):
			for j in range(len(self.current_state[i])):
				current_block_name = self.current_state[i][j]
				if j == 0 and self.goal_locations_by_name[self.current_state[i][j]][1] == 0:
					result.add(self.current_state[i][j])
					count = 1
					while j + count < len(self.current_state[i]):
						key = f"({i},{j + count})"
						if key in self.goal_locations_by_index:
							location_in_goal_state_of_current_block = self.goal_locations_by_name[self.current_state[i][j]]
							block_above_in_goal = self.goal_locations_by_index[f"({location_in_goal_state_of_current_block[0]},{j + count})"]
							block_above_in_current_state = self.current_state[i][j + count]
							if block_above_in_current_state == block_above_in_goal:
								result.add(self.current_state[i][j + count])
						count += 1
				else:
					pass
		return result

	def get_current_state_locations(self):
		self.moveable_blocks = set()
		for i in range(len(self.current_state)):
			if len(self.current_state[i]) > 0:
				if self.current_state[i][-1] not in self.non_moveable_blocks:
					self.moveable_blocks.add(self.current_state[i][-1])
				for j in range(len(self.current_state[i])):
					self.current_locations_by_name[self.current_state[i][j]] = (i, j)
					self.current_locations_by_index[f"({i},{j})"] = self.current_state[i][j]

	def determine_if_goal_state(self):
		return self.current_state == self.goal_state

	def is_table_block(self, block_name, block_location):
		if self.goal_locations_by_name[block_name][1] == 0:
			return True
		return False

	def is_on_table(self, block_location):
		if block_location[1] == 0:
			return True
		return False

	def determine_if_on_table(self, block_location):
		if block_location[1] == 0:
			return True
		return False

	def calculate_delta(self):
		if self.is_goal_state or self.is_initial_state:
			return 0
		total_delta = (len(self.goal_locations_by_name) - len(self.non_moveable_blocks)) * 10
		# for stack in range(len(self.current_state)):
		# 	for block in range(len(self.current_state[stack])):
		# 		current_block = self.current_state[stack][block]
		# 		block_specific_error = 0
		# 		if current_block not in self.non_moveable_blocks:
		# 			current_location = (stack, block)
		# 			if self.is_on_table(block_location=current_location):
		# 				if self.is_table_block(block_name=current_block, block_location=current_location):
		# 					self.non_moveable_blocks.add(current_block)
		# 				else:
		# 					# Current Block located on table but is not table block
		# 					block_specific_error += 5
		# 			else:
		# 				# Block is NOT a Table Block
		# 				if self.is_table_block(block_name=current_block, block_location=current_location):
		# 					# Current Block is NOT a Table Block based on Goal and is currently on the table
		# 					block_specific_error += 5
		# 				else:
		# 					# Current Block is NOT a Table block based on Goal and is NOT currently on the table
		# 					# Find out what SHOULD be located under the current block based on the Goal
		# 					location_of_current_block_in_goal_state = self.goal_locations_by_name[current_block]
		# 					under_block_in_goal_state = self.goal_locations_by_index[f"({location_of_current_block_in_goal_state[0]},{location_of_current_block_in_goal_state[1] - 1})"]
		# 					under_block_in_current_state = self.current_state[stack][block - 1]
		# 					if under_block_in_current_state == under_block_in_goal_state:
		# 						# Block located directly under current block is the same in the goal state
		# 						if under_block_in_current_state in self.non_moveable_blocks:
		# 							self.non_moveable_blocks.add(current_block)
		# 						else:
		# 							block_specific_error += 5
		# 					else:
		# 						block_specific_error += 5
		#
		# 		total_delta += (block_specific_error * 5)

		return total_delta


class BlockWorldAgent:
	def __init__(self):
		# If you want to do any initial processing, add it here.
		self.goal_state = None
		self.goal_arrangement = None
		self.visited_states = None
		self.goal_locations_by_name = None
		self.goal_locations_by_index = None
		self.state_queue = None

	def generate_next_states(self, some_state):
		possible_states = []
		if len(some_state.moveable_blocks) > 0:
			for moveable_block in some_state.moveable_blocks:
				current_arrangement = some_state.current_state
				current_block = moveable_block
				block_location = some_state.current_locations_by_name[current_block]
				if block_location[1] != 0:
					temp_arrangement = self.generate_table_move(block_location=block_location,
					                                            current_arrangement=current_arrangement)
					tuple_arrangement = tuple(tuple(i) for i in temp_arrangement if len(i) > 0)
					if tuple_arrangement not in self.visited_states:
						temp_state = State(current_state=tuple_arrangement,
						                   goal_state=self.goal_arrangement,
						                   goal_locations_by_name=self.goal_locations_by_name,
						                   goal_locations_by_index=self.goal_locations_by_index,
						                   non_moveable_blocks=set(b for b in some_state.non_moveable_blocks),
						                   number_of_moves=some_state.number_of_moves+1)
						temp_state.previously_visited = temp_state.current_state in self.visited_states
						temp_state.moves_used_to_get_to_state = [i for i in some_state.moves_used_to_get_to_state]
						temp_state.moves_used_to_get_to_state.append((f"{current_block}", "Table"))
						possible_states.append(temp_state)
						self.visited_states.add(temp_state.current_state)
				for stack in range(len(some_state.current_state)):
					temp_arrangement = [list(i) for i in some_state.current_state]
					if len(temp_arrangement[stack]) > 0:
						if temp_arrangement[stack][-1] == current_block:
							pass
						else:
							move_to_block = temp_arrangement[stack][-1]
							temp_arrangement[stack].append(temp_arrangement[block_location[0]].pop())
							tuple_arrangement = tuple(tuple(i) for i in temp_arrangement if len(i) > 0)
							if tuple_arrangement not in self.visited_states:
								temp_state = State(current_state=tuple_arrangement,
								                   goal_state=self.goal_arrangement,
								                   goal_locations_by_name=self.goal_locations_by_name,
								                   goal_locations_by_index=self.goal_locations_by_index,
								                   non_moveable_blocks=set(b for b in some_state.non_moveable_blocks),
								                   number_of_moves=some_state.number_of_moves+1)
								temp_state.previously_visited = temp_state.current_state in self.visited_states

								temp_state.moves_used_to_get_to_state = [i for i in some_state.moves_used_to_get_to_state]
								temp_state.moves_used_to_get_to_state.append((f"{current_block}", f"{move_to_block}"))
								possible_states.append(temp_state)
								self.visited_states.add(temp_state.current_state)

		return possible_states

	def generate_table_move(self, block_location, current_arrangement):
		new_arrangement = [list(i) for i in current_arrangement]
		new_arrangement.append([new_arrangement[block_location[0]].pop()])
		return new_arrangement

	def solve(self, initial_arrangement, goal_arrangement):
		# Add your code here! Your solve method should receive
		# as input two arrangements of blocks. The arrangements
		# will be given as lists of lists. The first item in each
		# list will be the bottom block on a stack, proceeding
		# upward. For example, this arrangement:
		#
		# [["A", "B", "C"], ["D", "E"]]
		#
		# ...represents two stacks of blocks: one with B on top
		# of A and C on top of B, and one with E on top of D.
		#
		# Your goal is to return a list of moves that will convert
		# the initial arrangement into the goal arrangement.
		# Moves should be represented as 2-tuples where the first
		# item in the 2-tuple is what block to move, and the
		# second item is where to put it: either on top of another
		# block or on the table (represented by the string "Table").
		#
		# For example, these moves would represent moving block B
		# from the first stack to the second stack in the example
		# above:
		#
		# ("C", "Table")
		# ("B", "E")
		# ("C", "A")
		self.visited_states = set()
		self.goal_locations_by_name = {}
		self.goal_locations_by_index = {}
		self.state_queue = PriorityQueue()
		for stack in range(len(goal_arrangement)):
			for i in range(len(goal_arrangement[stack])):
				self.goal_locations_by_name[goal_arrangement[stack][i]] = (stack, i)
				self.goal_locations_by_index[f"({stack},{i})"] = goal_arrangement[stack][i]
		self.goal_arrangement = tuple(tuple(i) for i in goal_arrangement if len(i) > 0)
		self.goal_state = State(current_state=self.goal_arrangement, goal_state=self.goal_arrangement,
		                        goal_locations_by_name=self.goal_locations_by_name,
		                        goal_locations_by_index=self.goal_locations_by_index)
		initial_state = State(current_state=tuple(tuple(i) for i in initial_arrangement if len(i) > 0), goal_state=self.goal_arrangement,
		                      goal_locations_by_name=self.goal_locations_by_name,
		                      goal_locations_by_index=self.goal_locations_by_index, is_initial_state=True,
		                      previously_visited=tuple(tuple(i) for i in initial_arrangement if len(i) > 0) in self.visited_states,
		                      number_of_moves=0)
		self.state_queue.put((initial_state.delta, initial_state))
		self.visited_states.add(initial_state.current_state)
		initial_state.previously_visited = initial_state.current_state in self.visited_states
		# temp_queue = PriorityQueue()
		is_solved = False
		while not is_solved:
			# if self.state_queue.empty():
			# 	self.state_queue = temp_queue
			# 	temp_queue = PriorityQueue()
			state_removed_from_queue = self.state_queue.get()
			next_states = self.generate_next_states(state_removed_from_queue[1])
			for i in range(len(next_states)):
				self.state_queue.put((next_states[i].delta, next_states[i]))
				# temp_queue.put((next_states[i].delta, next_states[i]))
				if next_states[i].determine_if_goal_state() or next_states[i].delta == 0:
					is_solved = True
					goal_state = next_states[i]
		print("\nNumber of Moves to solve: ", len(goal_state.moves_used_to_get_to_state))
		# return goal_state.moves_used_to_get_to_state, len(goal_state.moves_used_to_get_to_state)
		return goal_state.moves_used_to_get_to_state
