from queue import PriorityQueue


class State:
	def __init__(self, current_state, goal_state, goal_locations_by_name, goal_locations_by_index,
	             is_initial_state=False, number_of_moves=0):
		self.current_state = current_state
		self.is_initial_state = is_initial_state
		self.goal_state = goal_state
		self.number_of_moves = number_of_moves
		self.goal_locations_by_name = goal_locations_by_name
		self.goal_locations_by_index = goal_locations_by_index
		self.current_locations_by_name = {}
		self.current_locations_by_index = {}
		self.is_goal_state = self.determine_if_goal_state()
		self.moveable_blocks = None
		self.non_moveable_blocks = set()
		self.find_non_moveable()
		self.delta = self.calculate_delta()
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
		for i in range(len(self.current_state)):
			for j in range(len(self.current_state[i])):
				current_block = self.current_state[i][j]
				if current_block not in self.non_moveable_blocks:
					if j == 0 and self.goal_locations_by_name[self.current_state[i][j]][1] == 0:
						self.non_moveable_blocks.add(self.current_state[i][j])
						count = 1
						while j + count < len(self.current_state[i]):
							location_of_current_block_in_goal_state = self.goal_locations_by_name[current_block]
							key = f"({location_of_current_block_in_goal_state[0]},{j + count})"
							if key in self.goal_locations_by_index:
								block_above_in_goal = self.goal_locations_by_index[key]
								block_above_in_current_state = self.current_state[i][j + count]
								if block_above_in_current_state == block_above_in_goal:
									self.non_moveable_blocks.add(self.current_state[i][j + count])
							count += 1
					else:
						pass
		return

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

	def is_table_block(self, block_name):
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
		# Differences between current state and goal state
		total_delta = 0
		for i in range(len(self.current_state)):
			for j in range(len(self.current_state[i])):
				current_block = self.current_state[i][j]
				is_table_block = False
				# If is a table block but not on the table += 1
				# If Not a table block but is on table += 1
				if self.goal_locations_by_name[current_block][1] == 0:
					is_table_block = True
				if is_table_block:
					if j != 0:
						total_delta += 1
				else:
					# Not a table block
					if j == 0:
						total_delta += 1
					else:
						# Is Not Table Block, Check above and Below
						current_block_location_in_goal = self.goal_locations_by_name[current_block]
						# Check Below
						below_key = f"({current_block_location_in_goal[0]},{current_block_location_in_goal[1] - 1})"
						if below_key in self.goal_locations_by_index:
							block_below_in_goal = self.goal_locations_by_index[below_key]
							if j > 0:
								if self.current_state[i][j - 1] != block_below_in_goal:
									total_delta += 1
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

	def calculate_delta(self, current_state):
		# Differences between current state and goal state
		total_delta = 0
		for i in range(len(current_state)):
			for j in range(len(current_state[i])):
				current_block = current_state[i][j]
				is_table_block = False
				# If is a table block but not on the table += 1
				# If Not a table block but is on table += 1
				if self.goal_locations_by_name[current_block][1] == 0:
					is_table_block = True
				if is_table_block:
					if j != 0:
						total_delta += 1
				else:
					# Not a table block
					if j == 0:
						total_delta += 1
					else:
						# Is Not Table Block, Check above and Below
						current_block_location_in_goal = self.goal_locations_by_name[current_block]
						# Check Below
						below_key = f"({current_block_location_in_goal[0]},{current_block_location_in_goal[1] - 1})"
						if below_key in self.goal_locations_by_index:
							block_below_in_goal = self.goal_locations_by_index[below_key]
							if j > 0:
								if current_state[i][j - 1] != block_below_in_goal:
									total_delta += 1
		return total_delta

	def generate_next_states(self, some_state):
		possible_states = []
		if len(some_state.moveable_blocks) > 0:
			for block in some_state.moveable_blocks:
				added_block = False
				if block not in some_state.non_moveable_blocks:
					block_location = some_state.current_locations_by_name[block]
					# Only make a move to put ontop another block if other block is nonmoveable and it is correct spot
					possible_blocks_to_place_current_block_on = [_[-1] for _ in some_state.current_state if
					                                             _[-1] != block and _[
						                                             -1] in some_state.non_moveable_blocks]
					if len(possible_blocks_to_place_current_block_on) > 0:
						for base_block in possible_blocks_to_place_current_block_on:
							base_block_location_in_goal = self.goal_locations_by_name[base_block]
							base_block_location_in_current = some_state.current_locations_by_name[base_block]
							if len(some_state.goal_state[base_block_location_in_goal[0]]) > base_block_location_in_goal[
								1] + 1:
								key = f"({base_block_location_in_goal[0]},{base_block_location_in_goal[1] + 1})"
								if key in some_state.goal_locations_by_index:
									if some_state.goal_locations_by_index[key] == block:
										# Then Move the block ontop
										temp_arrangement = [list(i) for i in some_state.current_state]
										temp_arrangement[base_block_location_in_current[0]].append(
											temp_arrangement[block_location[0]].pop())
										tuple_arrangement = tuple(tuple(i) for i in temp_arrangement if len(i) > 0)
										if tuple_arrangement not in self.visited_states:
											temp_state = State(current_state=tuple_arrangement,
											                   goal_state=self.goal_arrangement,
											                   goal_locations_by_name=self.goal_locations_by_name,
											                   goal_locations_by_index=self.goal_locations_by_index,
											                   number_of_moves=some_state.number_of_moves + 1)
											temp_state.non_moveable_blocks.add(block)
											temp_state.previously_visited = temp_state.current_state in self.visited_states
											temp_state.moves_used_to_get_to_state = [i for i in
											                                         some_state.moves_used_to_get_to_state]
											temp_state.moves_used_to_get_to_state.append((f"{block}", f"{base_block}"))
											possible_states.append(temp_state)
											self.visited_states.add(temp_state.current_state)
									else:
										if block_location[1] != 0:
											temp_arrangement = [list(i) for i in some_state.current_state]
											temp_arrangement.append([temp_arrangement[block_location[0]].pop()])
											tuple_arrangement = tuple(tuple(i) for i in temp_arrangement if len(i) > 0)
											if tuple_arrangement not in self.visited_states:
												temp_state = State(current_state=tuple_arrangement,
												                   goal_state=self.goal_arrangement,
												                   goal_locations_by_name=self.goal_locations_by_name,
												                   goal_locations_by_index=self.goal_locations_by_index,
												                   number_of_moves=some_state.number_of_moves + 1)
												current_blocks_location = temp_state.current_locations_by_name[block]
												current_blocks_location_in_goal = self.goal_locations_by_name[block]
												if current_blocks_location[1] == current_blocks_location_in_goal[1]:
													temp_state.non_moveable_blocks.add(block)

												temp_state.previously_visited = temp_state.current_state in self.visited_states
												temp_state.moves_used_to_get_to_state = [i for i in
												                                         some_state.moves_used_to_get_to_state]
												temp_state.moves_used_to_get_to_state.append((f"{block}", "Table"))
												possible_states.append(temp_state)
												self.visited_states.add(temp_state.current_state)
												added_block = True
						if not added_block:
							if block_location[1] != 0:
								temp_arrangement = [list(i) for i in some_state.current_state]
								temp_arrangement.append([temp_arrangement[block_location[0]].pop()])
								tuple_arrangement = tuple(tuple(i) for i in temp_arrangement if len(i) > 0)
								if tuple_arrangement not in self.visited_states:
									temp_state = State(current_state=tuple_arrangement,
									                   goal_state=self.goal_arrangement,
									                   goal_locations_by_name=self.goal_locations_by_name,
									                   goal_locations_by_index=self.goal_locations_by_index,
									                   number_of_moves=some_state.number_of_moves + 1)
									current_blocks_location = temp_state.current_locations_by_name[block]
									current_blocks_location_in_goal = self.goal_locations_by_name[block]
									if current_blocks_location[1] == current_blocks_location_in_goal[1]:
										temp_state.non_moveable_blocks.add(block)

									temp_state.previously_visited = temp_state.current_state in self.visited_states
									temp_state.moves_used_to_get_to_state = [i for i in
									                                         some_state.moves_used_to_get_to_state]
									temp_state.moves_used_to_get_to_state.append((f"{block}", "Table"))
									possible_states.append(temp_state)
									self.visited_states.add(temp_state.current_state)
					else:
						if block_location[1] != 0:
							temp_arrangement = [list(i) for i in some_state.current_state]
							temp_arrangement.append([temp_arrangement[block_location[0]].pop()])
							tuple_arrangement = tuple(tuple(i) for i in temp_arrangement if len(i) > 0)
							if tuple_arrangement not in self.visited_states:
								temp_state = State(current_state=tuple_arrangement,
								                   goal_state=self.goal_arrangement,
								                   goal_locations_by_name=self.goal_locations_by_name,
								                   goal_locations_by_index=self.goal_locations_by_index,
								                   number_of_moves=some_state.number_of_moves + 1)
								current_blocks_location = temp_state.current_locations_by_name[block]
								current_blocks_location_in_goal = self.goal_locations_by_name[block]
								if current_blocks_location[1] == current_blocks_location_in_goal[1]:
									temp_state.non_moveable_blocks.add(block)

								temp_state.previously_visited = temp_state.current_state in self.visited_states
								temp_state.moves_used_to_get_to_state = [i for i in
								                                         some_state.moves_used_to_get_to_state]
								temp_state.moves_used_to_get_to_state.append((f"{block}", "Table"))
								possible_states.append(temp_state)
								self.visited_states.add(temp_state.current_state)
		return possible_states

	def generate_table_move(self, block_location, current_arrangement):
		new_arrangement = [list(i) for i in current_arrangement]
		new_arrangement.append([new_arrangement[block_location[0]].pop()])
		return tuple(tuple(arr) for arr in new_arrangement)

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
		self.goal_arrangement = tuple(tuple(i) for i in goal_arrangement)
		self.goal_state = State(current_state=self.goal_arrangement, goal_state=self.goal_arrangement,
		                        goal_locations_by_name=self.goal_locations_by_name,
		                        goal_locations_by_index=self.goal_locations_by_index)
		initial_state = State(current_state=tuple(tuple(i) for i in initial_arrangement if len(i) > 0),
		                      goal_state=self.goal_arrangement,
		                      goal_locations_by_name=self.goal_locations_by_name,
		                      goal_locations_by_index=self.goal_locations_by_index, is_initial_state=True)
		self.state_queue.put((initial_state.delta, initial_state))
		self.visited_states.add(initial_state.current_state)
		is_solved = False
		while not is_solved:
			if self.state_queue.empty():
				return state_removed_from_queue.moves_used_to_get_to_state
			state_removed_from_queue = self.state_queue.get()
			next_states = self.generate_next_states(state_removed_from_queue[1])
			for i in range(len(next_states)):
				self.state_queue.put((next_states[i].delta, next_states[i]))
				if next_states[i].determine_if_goal_state() or next_states[i].delta == 0:
					is_solved = True
					goal_state = next_states[i]
		return goal_state.moves_used_to_get_to_state
