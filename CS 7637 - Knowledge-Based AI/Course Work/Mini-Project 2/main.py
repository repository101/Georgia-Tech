from BlockWorldAgent import BlockWorldAgent



"""
Pseudocode: BlockWorldAgent.solve

def solve(self, initial_arrangement, goal_arrangement):
    Initialize 'Goal_State' using goal_arrangement
    Initialize 'Initial_State' using initial_arrangement
    Initialize 'States' Priority Queue to maintain the flow to lower delta states
    Initialize 'Visited_State' set, to track what states have been visited 
    Add 'Initial_State' to the 'States' Priority Queue
    Solved:= False
    
    While the problem is not solved and the States Priority Queue is not empty:
        Current_State:= Get lowest delta state from 'States' Priority Queue
        Next_States:= Generate next possible states using Current_State
        Iterate over the returned Next_States:
            Add each 'temp_state' to the 'States' Priority Queue
            If 'temp_state' == 'Goal_State':
                Solved:= True
                return the variable within the 'temp_state' which tracks the moves used to reach that state

def generate_next_states(some_state):
    returned_states:= set()
    For 'temp_block' in some_state.moveable_blocks:
        Find 'all_locations' where 'temp_block' could move
        For 'location' in 'all_locations':
            If 'location' non moveable and 'temp_block' should be on top:
                Create New State with 'temp_block' at new location
                Add the new state to 'returned_states'
        If no valid moves were made and 'temp_block' is not currently located on the table:
            Create New State with 'temp_block' on the table
            Add the new state to 'returned_states'
    return returned_states
    
    
Pseudocode: State

class State:
    def __init__(all needed parameters):
        Set Current State
        Set Goal State
        Calculate current states delta
        Find blocks which can be moved
        Find blocks which cannot be moved

"""




def test():
    # This will test your BlockWorldAgent
    # with eight initial test cases.
    test_agent = BlockWorldAgent()
    # tests = [{"Name": "Test 1", "Initial": [['A', 'B', 'C'], ['D', 'E']], "Goal": [['A', 'C'], ['D', 'E', 'B']]},
    #          {"Name": "Test 2", "Initial": [['D', 'I'], ['G'], ['F', 'H'], ['J', 'C', 'A'], ['K'], ['B', 'E']], "Goal": [['G', 'A'], ['I', 'D', 'E', 'J', 'H', 'K'], ['B'], ['F', 'C']]},
    #          {"Name": "Test 3", "Initial": [['L'], ['G', 'F', 'C', 'B'], ['I', 'E'], ['M', 'N', 'J', 'H', 'A'], ['K', 'D']], "Goal": [['A'], ['C', 'N', 'K', 'H', 'B', 'I'], ['M', 'J'], ['F', 'E', 'D', 'G'], ['L']]},
    #          {"Name": "Test 4", "Initial": [['C'], ['M', 'D', 'K', 'B'], ['F', 'A', 'I', 'H', 'G', 'E', 'L'], ['J', 'N']], "Goal": [['F', 'M'], ['K', 'D', 'L'], ['J', 'C', 'N', 'G', 'B', 'A', 'E', 'I'], ['H']]},
    #          {"Name": "Test 5", "Initial": [['H', 'G', 'E', 'C'], ['K', 'A', 'I', 'B', 'D'], ['J', 'L'], ['F']], "Goal": [['E', 'C'], ['F', 'L', 'B', 'I', 'A'], ['G', 'H'], ['K', 'D'], ['J']]},
    #          {"Name": "Test 6", "Initial": [['D', 'B', 'M', 'C'], ['A', 'F', 'J'], ['O', 'E', 'H', 'N', 'K', 'I'], ['P'], ['G'], ['L']], "Goal": [['O', 'I', 'E'], ['N', 'J', 'P', 'A', 'K'], ['L', 'F', 'G', 'D', 'M'], ['B', 'C'], ['H']]},
    #          {"Name": "Test 7", "Initial": [['H'], ['P', 'I', 'M'], ['O', 'D', 'C', 'B', 'G', 'F', 'E'], ['A'], ['K'], ['L', 'N'], ['J']], "Goal": [['P', 'A', 'O'], ['I', 'N', 'L', 'D'], ['M', 'E'], ['J', 'B', 'K', 'C', 'H', 'F', 'G']]},
    #          {"Name": "Test 8", "Initial": [['B', 'F', 'J', 'G', 'M'], ['R', 'D', 'Q', 'N', 'E'], ['H'], ['C', 'P', 'A'], ['O', 'K'], ['I'], ['L']], "Goal": [['Q', 'M', 'L', 'N', 'C'], ['B', 'P', 'G', 'I', 'J'], ['A', 'F', 'K'], ['E', 'H', 'D', 'O', 'R']]},
    #          {"Name": "Test 9", "Initial": [['B', 'K', 'D', 'R', 'L', 'I', 'P'], ['E', 'G', 'O'], ['S', 'M', 'N', 'H', 'C'], ['T', 'Q'], ['A', 'J', 'F']], "Goal": [['B'], ['M', 'S', 'A', 'C', 'P'], ['F', 'L', 'H', 'D'], ['O', 'R', 'J'], ['N', 'Q', 'E', 'K'], ['G', 'T', 'I']]},
    #          {"Name": "Test 10", "Initial": [['L', 'N', 'M', 'I', 'O', 'Q', 'J'], ['A', 'T', 'H', 'F', 'P'], ['K', 'D', 'S', 'C', 'R', 'E', 'B'], ['G']], "Goal": [['E', 'D', 'C', 'T', 'P', 'K', 'M'], ['O', 'J'], ['L', 'B', 'I'], ['S', 'Q'], ['F', 'A'], ['N', 'H', 'G', 'R']]},
    #          {"Name": "Test 11", "Initial": [['K', 'B', 'H', 'C', 'U'], ['V', 'G', 'M'], ['N', 'T', 'O', 'F', 'W'], ['S', 'A', 'P', 'J', 'L'], ['I'], ['E', 'R', 'Q'], ['D']], "Goal": [['C', 'G'], ['K', 'N'], ['L', 'D', 'H', 'R', 'P'], ['E', 'S', 'I'], ['B', 'J', 'F', 'V'], ['W', 'A', 'Q', 'M', 'U', 'O', 'T']]},
    #          {"Name": "Test 12", "Initial": [['A', 'B', 'C'], ['D', 'E']], "Goal": [['A', 'B', 'C', 'D', 'E']]},
    #          {"Name": "Test 13", "Initial": [['C', 'M', 'I', 'B', 'R', 'H', 'K', 'G'], ['D', 'F', 'A', 'U', 'P', 'T', 'O', 'S'], ['E'], ['J', 'L'], ['N'], ['Q']], "Goal": [['P', 'R', 'A', 'M'], ['U'], ['B', 'N', 'Q'], ['F', 'H', 'G', 'D', 'L', 'T', 'J', 'K', 'S', 'I', 'O', 'C', 'E']]},
    #          {"Name": "Test 14", "Initial": [['A', 'B', 'C'], ['D', 'E']], "Goal": [['D', 'E', 'A', 'B', 'C']]},
    #          {"Name": "Test 15", "Initial": [['A', 'B', 'C'], ['D', 'E']], "Goal": [['C', 'D'], ['E', 'A', 'B']]},
    #          {"Name": "Test 16", "Initial": [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']], "Goal": [['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']]},
    #          {"Name": "Test 17", "Initial": [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']], "Goal": [['I', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']]},
    #          {"Name": "Test 18", "Initial": [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']], "Goal": [['H', 'E', 'F', 'A', 'C'], ['B', 'D'], ['G', 'I']]},
    #          {"Name": "Test 19", "Initial": [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']], "Goal": [['F', 'D', 'C', 'I', 'G', 'A'], ['B', 'E', 'H']]},
    #          {"Name": "Test 20", "Initial": [['I', 'F', 'L', 'K', 'G', 'H', 'E', 'J'], ['M'], ['A', 'B'], ['C'], ['D']], "Goal": [['C', 'G', 'I', 'E', 'K', 'J'], ['L', 'F', 'B', 'H'], ['M', 'A', 'D']]}]
    #
    # for i in tests:
    #     print("Current Test: \n\t", i["Name"])
    #     initial_arrangement_1 = i["Initial"]
    #     if i["Name"] == "Test 3":
    #         print()
    #     # initial_arrangement_1 = [['A', 'B'], ['D', 'E'], ['C']]
    #     goal_arrangement_1 = i["Goal"]
    #     results, testy = test_agent.solve(initial_arrangement_1, goal_arrangement_1)
    #     goal_set = set()
    #     goal_set.add(tuple(tuple(k) for k in goal_arrangement_1))
    #     test_tuple = tuple(tuple(k) for k in testy)
    #     if test_tuple not in goal_set:
    #         print("\tFailed: ", i["Name"])
    #     else:
    #         print("\tSuccess")
    # print()

    initial_arrangement_1 = [["A", "B", "C"], ["D", "E"]]
    goal_arrangement_1 = [["A", "C"], ["D", "E", "B"]]
    goal_arrangement_2 = [["A", "B", "C", "D", "E"]]
    goal_arrangement_3 = [["D", "E", "A", "B", "C"]]
    goal_arrangement_4 = [["C", "D"], ["E", "A", "B"]]

    print(test_agent.solve(initial_arrangement_1, goal_arrangement_1))
    print(test_agent.solve(initial_arrangement_1, goal_arrangement_2))
    print(test_agent.solve(initial_arrangement_1, goal_arrangement_3))
    print(test_agent.solve(initial_arrangement_1, goal_arrangement_4))

    initial_arrangement_2 = [["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"]]
    goal_arrangement_5 = [["A", "B", "C", "D", "E", "F", "G", "H", "I"]]
    goal_arrangement_6 = [["I", "H", "G", "F", "E", "D", "C", "B", "A"]]
    goal_arrangement_7 = [["H", "E", "F", "A", "C"], ["B", "D"], ["G", "I"]]
    goal_arrangement_8 = [["F", "D", "C", "I", "G", "A"], ["B", "E", "H"]]

    print(test_agent.solve(initial_arrangement_2, goal_arrangement_5))
    print(test_agent.solve(initial_arrangement_2, goal_arrangement_6))
    print(test_agent.solve(initial_arrangement_2, goal_arrangement_7))
    print(test_agent.solve(initial_arrangement_2, goal_arrangement_8))

    # NON OPTIMAL TESTS
    # Number of Noves should be 12 or less
    limit = 12
    initial_arrangement_1 = [['D'], ['B', 'E', 'C', 'F', 'G', 'H', 'A'], ['I']]
    goal_arrangement_1 = [['I', 'B', 'D', 'E'], ['A'], ['F', 'C', 'G', 'H']]
    result = test_agent.solve(initial_arrangement_1, goal_arrangement_1)
    print(result)
    length = len(result)
    if length > 12:
        print("\t NON Optimal Test 1: FAILED")
        print(f"\t Number of Moves to solve: {length} exceeded the limit of {limit}")
    else:
        print("Success!")

    # Should be less than 18
    limit = 18
    initial_arrangement_1 = [['K', 'I', 'M', 'L', 'J', 'B', 'H', 'D', 'E'], ['F'], ['A'], ['C', 'G']]
    goal_arrangement_1 = [['M', 'K', 'I', 'D'], ['C', 'E'], ['A'], ['B', 'J', 'G', 'L', 'F', 'H']]
    result = test_agent.solve(initial_arrangement_1, goal_arrangement_1)
    print(result)
    length = len(result)
    if length > 18:
        print("\t NON Optimal Test 2: FAILED")
        print(f"\t Number of Moves to solve: {length} exceeded the limit of {limit}")
    else:
        print("Success!")
    #
    # Should be less than 26
    limit = 26
    initial_arrangement_1 = [['E', 'D', 'J'], ['A', 'C'], ['I', 'G', 'H', 'B', 'M'], ['K'], ['N', 'P', 'O', 'L'], ['Q', 'F']]
    goal_arrangement_1 = [['M', 'J', 'O', 'I', 'K', 'L', 'D', 'H', 'N', 'P', 'A', 'C', 'F'], ['G', 'E', 'Q', 'B']]
    result = test_agent.solve(initial_arrangement_1, goal_arrangement_1)
    print(result)
    length = len(result)
    if length > 26:
        print("\t NON Optimal Test 3: FAILED")
        print(f"\t Number of Moves to solve: {length} exceeded the limit of {limit}")
    else:
        print("Success!")

    #
    # # Should be less than 20
    limit = 20
    initial_arrangement_1 = [['G', 'D', 'K', 'B', 'H'], ['L', 'J', 'M', 'C'], ['A', 'E', 'F', 'N'], ['I'], ['O']]
    goal_arrangement_1 = [['L', 'C', 'B', 'A', 'J', 'H', 'O', 'G', 'I'], ['K', 'D', 'M'], ['E'], ['N'], ['F']]
    result = test_agent.solve(initial_arrangement_1, goal_arrangement_1)
    print(result)
    length = len(result)
    if length > 20:
        print("\t NON Optimal Test 4: FAILED")
        print(f"\t Number of Moves to solve: {length} exceeded the limit of {limit}")
    else:
        print("Success!")
    #
    # Should be less than 30
    limit = 30
    initial_arrangement_1 = [['J', 'N', 'S', 'I', 'K', 'Q', 'F', 'P'], ['E', 'G', 'B', 'M', 'D', 'A'], ['C', 'O', 'H'], ['L', 'R']]
    goal_arrangement_1 = [['L', 'N', 'I', 'C', 'J', 'O'], ['P', 'Q'], ['S', 'K', 'R', 'D', 'M', 'G', 'A', 'H'], ['B', 'E', 'F']]
    result = test_agent.solve(initial_arrangement_1, goal_arrangement_1)
    print(result)
    length = len(result)
    if length > 30:
        print("\t NON Optimal Test 5: FAILED")
        print(f"\t Number of Moves to solve: {length} exceeded the limit of {limit}")
    else:
        print("Success!")

    # print()

if __name__ == "__main__":
    test()