from SentenceReadingAgent import SentenceReadingAgent


def test():
	# This will test your SentenceReadingAgent
	# with nine initial test cases.
	
	test_agent = SentenceReadingAgent()
	sentences = dict(
		sentence_1="Ada brought a short note to Irene.",
		sentence_2="David and Lucy walk one mile to go to school every day at 8:00AM when there is no snow.",
		sentence_3="There are a thousand children in this town.",
		sentence_4="Serena and Ada took the blue rock to the street.",
		sentence_5="The white dog and the blue horse play together.",
		sentence_6="This year David will watch a play.",
		sentence_7="There are three men in the car.",
		sentence_8="Give us all your money.",
		sentence_9="She will write him a love letter.",
		entence_10="There are one hundred adults in that city.",
		sentence_11="Serena ran a mile this morning.",
		sentence_12="Serena and Ada took the blue rock to the street.",
		sentence_13="This year will be the best one yet.",
		sentence_14="Bring the dog to the other room.",
		sentence_15="Serena ran a mile this morning.",
		sentence_16="The water is blue.",
		sentence_17="The red fish is in the river.",
		sentence_18="It will snow soon.",
		sentence_19="Their children are in school.",
		sentence_20="The red fish is in the river.",
		sentence_21="My dog Red is very large.",
		sentence_22="Their children are in school.",
		sentence_23="The red fish is in the river.",
		sentence_24="The water is blue.",
		sentence_25="The island is east of the city.",
		sentence_26="The blue bird will sing in the morning.",
		sentence_27="There is snow at the top of the mountain.",
		sentence_28="This tree came from the island.",
		sentence_29="She told her friend a story.",
		sentence_30="She will write him a love letter.",
		sentence_31="Serena saw a home last night with her friend.",
		sentence_32="She told her friend a story.",
		sentence_33="The house is made of paper.",
		sentence_34="Bring the box to the other room.",
		sentence_35="Watch your step.",
		sentence_36="Lucy will write a book.",
		sentence_37="The blue bird will sing in the morning.",
		sentence_38="Frank took the horse to the farm.",
		sentence_39="A tree is made of wood.",
		sentence_40="There are one hundred adults in that city.")

	questions = dict(question_40_1="Who is in the city?",
	                 question_39_1="What is a tree made of?",
	                 question_38_1="What did Frank take to the farm?",
	                 question_37_1="What will sing in the morning?",
	                 question_36_1={"q0": "What will Lucy write?",
	                                "q1": "What will Lucy do?"},
	                 question_35_1="What should you watch?",
	                 question_34_1="What should be brought to the other room?",
	                 question_33_1="What is the house made of?",
	                 question_32_1="Who was told a story?",
	                 question_31_1="Who was with Serena?",
	                 question_30_1="Who was written a love letter?",
	                 question_29_1="What did she tell?",
	                 question_28_1="What came from the island?",
	                 question_27_1={"q0": "What is on top of the mountain?",
	                                "q1": "Where is the snow?"},
	                 question_26_1="What will sing in the morning?",
	                 question_25_1="Where is the island?",
	                 question_24_1="What color is the water?",
	                 question_23_1="Where is the fish?",
	                 question_22_1="Where are their children?",
	                 question_21_1={"q0": "What is my dog's name?",
	                                "q1": "How big is my dog?",
	                                "q2": "What animal is Red?"},
	                 question_20_1="What is in the river?",
	                 question_19_1="Who is in school?",
	                 question_18_1="When will it snow?",
	                 question_17_1="What color is the fish?",
	                 question_16_1="What is blue?",
	                 question_15_1="What did Serena run?",
	                 question_14_1="What should be brought to the other room?",
	                 question_13_1="What will this year be?",
	                 question_12_1={"q0": "What color was the rock?",
	                                "q1": "What was blue?"},
	                 question_11_1="When did Serena run?",
	                 question_10_1="How many adults are in this city?",
	                 question_9_1={"q0": "Who wrote a love letter?",
	                               "q1": "What will she write him?"},
	                 question_8_1={"q0": "How much of your money should you give us?",
	                               "q1": "What should you give us?"},
	                 question_7_1="Where are the men?",
	                 question_6_1="What will David watch?",
	                 question_5_1={"q0": "What color is the horse?",
	                               "q1": "What animal is white?"},
	                 question_4_1="Where did they take the rock?",
	                 question_3_1="Where are the children?",
	                 question_2_1={"q0": "Who does Lucy go to school with?", "q1": "Where do David and Lucy go?",
	                               "q2": "How far do David and Lucy walk?",
	                               "q3": "How do David and Lucy get to school?",
	                               "q4": "At what time do David and Lucy walk to school?"},
	                 question_1_1={"q0": "Who brought the note?", "q1": "What did Ada bring?",
	                               "q2": "Who did Ada bring the note to?", "q3": "How long was the note?"})
	
	answers = \
		dict(question_40_1=["adults"],
		     question_39_1=["wood"],
		     question_38_1=["horse", "the horse"],
		     question_37_1=["bird", "blue bird"],
		     question_36_1={"q0": ["book"],
		                    "q1": ["write"]},
		     question_35_1=["step"],
		     question_34_1=["the box"],
		     question_33_1=["paper"],
		     question_32_1=["her friend"],
		     question_31_1=["her friend"],
		     question_30_1=["him"],
		     question_29_1=["a story", "story"],
		     question_28_1=["tree", "this tree"],
		     question_27_1={"q0": "snow", "q1": ["mountain", "Mountain"]},
		     question_26_1=["bird", 'blue bird'],
		     question_25_1=["east of the city", "east"],
		     question_24_1=["blue"],
		     question_23_1=["river", "the river"],
		     question_22_1=["school", "in school"],
		     question_21_1={"q0": ["red", "Red"],
		                    "q1": ["large", "very large"],
		                    "q2": ["dog", "a dog"]},
		     question_20_1=["fish", "red fish"],
		     question_19_1=["children", "their children"],
		     question_18_1=["soon"],
		     question_17_1=["red"],
		     question_16_1=["water", "the water"],
		     question_15_1=["a mile", "mile"],
		     question_14_1=["dog", "the dog"],
		     question_13_1=["the best", "best", "best one yet"],
		     question_12_1={"q0": ["blue"],
		                    "q1": ["rock", "the rock"]},
		     question_11_1=["morning", "this morning"],
		     question_10_1=["one hundred", "hundred"],
		     question_9_1={"q0": ["she", "She"],
		                   "q1": ["letter", "love letter", "love letter"]},
		     question_8_1={"q0": ["all"],
		                   "q1": ["money"]},
		     question_7_1=["car", "the car"],
		     question_6_1=["a play", "play"],
		     question_5_1={"q0": ["blue"],
		                   "q1": ["dog"]},
		     question_4_1=["the street", "street"],
		     question_3_1=["town", "this town"],
		     question_2_1={"q0": ["David", "david"], "q1": ["school"],
		                   "q2": ["mile", "a mile", "one mile"], "q3": ["walk"],
		                   "q4": ["8:00AM"]},
		     question_1_1={"q0": ["Ada", "ada"], "q1": ["note", "a note"],
		                   "q2": ["Irene", "irene"], "q3": ["short"]})
	
	# print(test_agent.solve(sentences["sentence_8"], questions["question_8_1"]["q1"]))
	
	correct = 0
	num_questions = 0
	for key, val in sentences.items():
		question_key = "question_" + key.split("_")[-1] + "_1"
		temp_sentence = val
		if question_key in questions:
			if isinstance(questions[question_key], dict):
				for key_1, val_1 in questions[question_key].items():
					num_questions += 1
					result = test_agent.solve(temp_sentence, val_1)
					if len(answers[question_key]) > 0:
						if result not in answers[question_key][key_1]:
							print("Sentence: ")
							print(f"\t{val}")
							print("Question: ")
							print(f"\t{questions[question_key][key_1]}")
							print(f"\t\tReturned Answer: {result}")
							print(f"\t\tExpected Answer: {answers[question_key][key_1]}")
						else:
							correct += 1
			else:
				num_questions += 1
				temp_question = questions[question_key]
				result = test_agent.solve(temp_sentence, temp_question)
				if len(answers[question_key]) > 0:
					if result not in answers[question_key]:
						print("Sentence: ")
						print(f"\t{val}")
						print("Question: ")
						print(f"\t{questions[question_key]}")
						print(f"\t\tReturned Answer: {result}")
						print(f"\t\tExpected Answer: {answers[question_key]}")
					else:
						correct += 1
	print(f"Total Correct: {correct} out of {num_questions}")


# sentence_1 = "Ada brought a short note to Irene."
# question_1 = "Who brought the note?"
# question_2 = "What did Ada bring?"
# question_3 = "Who did Ada bring the note to?"
# question_4 = "How long was the note?"
#
# sentence_2 = "David and Lucy walk one mile to go to school every day at 8:00AM when there is no snow."
# sentence_3 = "There are a thousand children in this town."
# sentence_4 = "Serena and Ada took the blue rock to the street."
# sentence_5 = "The white dog and the blue horse play together."
# sentence_6 = "This year David will watch a play."
# sentence_7 = "There are three men in the car."
# sentence_8 = "Give us all your money."
# sentence_9 = "She will write him a love letter."
# sentence_10 = "There are one hundred adults in that city."
# sentence_11 = "Serena ran a mile this morning."
# sentence_12 = "Serena and Ada took the blue rock to the street."
# sentence_13 = "This year will be the best one yet."
# sentence_14 = "Bring the dog to the other room."
# sentence_15 = "Serena ran a mile this morning."
# sentence_16 = "The water is blue."
# sentence_17 = "The red fish is in the river."
# sentence_18 = "It will snow soon."
# sentence_19 = "Their children are in school."
# sentence_20 = "The red fish is in the river."
# sentence_21 = "My dog Red is very large."
# sentence_22 = "Their children are in school."
# sentence_23 = "The red fish is in the river."
# sentence_24 = "The water is blue."
# sentence_25 = "The island is east of the city."
# sentence_26 = "The blue bird will sing in the morning."
# sentence_27 = "There is snow at the top of the mountain."
#
# question_27_1 = "What is on top of the mountain?"
# question_26_1 = "What will sing in the morning?"
# question_25_1 = "Where is the island?"
# question_24_1 = "What color is the water?"
# question_23_1 = "Where is the fish?"
# question_22_1 = "Where are their children?"
# question_21_1 = "What is my dog's name?"
# question_20_1 = "What is in the river?"
# question_19_1 = "Who is in school?"
# question_18_1 = "When will it snow?"
# question_17_1 = "What color is the fish?"
# question_16_1 = "What is blue?"
# question_15_1 = "What did Serena run?"
# question_14_1 = "What should be brought to the other room?"
# question_13_1 = "What will this year be?"
# question_12_1 = "What color was the rock?"
# question_11_1 = "When did Serena run?"
# question_10_1 = "How many adults are in this city?"
# question_9_1 = "Who wrote a love letter?"
# question_8_1 = "How much of your money should you give us?"
# question_7_1 = "Where are the men?"
# question_6_1 = "What will David watch?"
# question_3_1 = "Where are the children?"
# question_4_1 = "Where did they take the rock?"
# question_5_1 = "What color is the horse?"
# question_5 = "Who does Lucy go to school with?"
# question_6 = "Where do David and Lucy go?"
# question_7 = "How far do David and Lucy walk?"
# question_8 = "How do David and Lucy get to school?"
# question_9 = "At what time do David and Lucy walk to school?"
#
# question_10 = "Was there snow when they walk to school?"
# question_11 = "Did they walk in the morning or afternoon?"
# question_12 = "How often did they walk to school when there was snow?"
# question_13 = "What did David do?"
# question_14 = "Why did Lucy do this?"
# question_15 = "Where does David go?"
# question_17 = "How far does David walk?"
# question_18 = "When would they walk?"
# question_19 = "When would David walk?"
# question_20 = "When would Lucy walk?"
# question_22 = "Was the walk in the morning or afternoon?"
# question_23 = "Was the walk in the morning?"
# question_24 = "Was the walk in the afternoon?"
# question_25 = "Who was with Lucy?"
# question_26 = "Who was with David?"
# question_27 = "What time was it?"
# question_28 = "What is the reason they walk to school?"
# question_29 = "Where does Lucy go?"
# question_30 = "Where do they go?"
# question_31 = "David and Lucy walk a mile to go to school every day at 8:00AM, when?"
# question_32 = "Did they walk in the morning?"
# question_33 = "Did they walk in the afternoon?"
# question_34 = "Did they walk to school every day?"
#
# print(test_agent.solve(sentence_3, question_3_1))  # "town"
# print(test_agent.solve(sentence_4, question_4_1))  # "street"
# print(test_agent.solve(sentence_5, question_5_1))  # "blue"
# print(test_agent.solve(sentence_6, question_6_1))  # "play"
# print(test_agent.solve(sentence_7, question_7_1))  # "car"
# print(test_agent.solve(sentence_8, question_8_1))  # "all"
# print(test_agent.solve(sentence_9, question_9_1))  # "she"
# print(test_agent.solve(sentence_10, question_10_1))  # "one hundred"
# print(test_agent.solve(sentence_11, question_11_1))  # "morning" or "this morning"
# print(test_agent.solve(sentence_12, question_12_1))  # "blue"
# print(test_agent.solve(sentence_13, question_13_1))  # "the best"
# print(test_agent.solve(sentence_14, question_14_1))  # "the dog"
# print(test_agent.solve(sentence_15, question_15_1))  # "a mile"
# print(test_agent.solve(sentence_16, question_16_1))  # "water"
# print(test_agent.solve(sentence_17, question_17_1))  # "red"
# print(test_agent.solve(sentence_18, question_18_1))  # "soon"
# print(test_agent.solve(sentence_19, question_19_1))  # "children"
# print(test_agent.solve(sentence_20, question_20_1))  # "fish"
# print(test_agent.solve(sentence_21, question_21_1))  # "Red"
# print(test_agent.solve(sentence_22, question_22_1))  # "school"
# print(test_agent.solve(sentence_23, question_23_1))  # "the river"
# print(test_agent.solve(sentence_24, question_24_1))  # "blue"
# print(test_agent.solve(sentence_25, question_25_1))  # "east" or "east of the city"
# print(test_agent.solve(sentence_26, question_26_1))  # "bird" or "blue bird"
# print(test_agent.solve(sentence_27, question_27_1))  # "snow"


#

# # Who Group
# print("\nWHO GROUP")
# print(test_agent.solve(sentence_1, question_1))  # "Ada"
# print(test_agent.solve(sentence_1, question_3))  # "Irene"
# print(test_agent.solve(sentence_2, question_5))  # "David"
# print(test_agent.solve(sentence_2, question_25))  # "David"
# print(test_agent.solve(sentence_2, question_26))  # "Lucy"
#
# # What Group
# print("\nWHAT GROUP")
# print(test_agent.solve(sentence_1, question_2))  # "note" or "a note"
# print(test_agent.solve(sentence_2, question_9))  # "8:00AM"
# print(test_agent.solve(sentence_2, question_13))  # "walk"
# print(test_agent.solve(sentence_2, question_27))  # "8:00AM"
#
# # When Group
# print(test_agent.solve(sentence_2, question_18))  # "8:00AM""
# print(test_agent.solve(sentence_2, question_19))  # "8:00AM""
# print(test_agent.solve(sentence_2, question_20))  # "8:00AM""
# print(test_agent.solve(sentence_2, question_31))  # "snow""
#
# # Where Group
# print(test_agent.solve(sentence_2, question_6))  # "school"
# print(test_agent.solve(sentence_2, question_15))  # "school""
# print(test_agent.solve(sentence_2, question_29))  # "school""
# print(test_agent.solve(sentence_2, question_30))  # "school""
#
# # Why Group
# print(test_agent.solve(sentence_2, question_14))  # "snow"
#
# # How Group
# print(test_agent.solve(sentence_1, question_4))  # "short"
# print(test_agent.solve(sentence_2, question_7))  # "mile" or "a mile"
# print(test_agent.solve(sentence_2, question_8))  # "walk"
# print(test_agent.solve(sentence_2, question_12))  # "every day"
# print(test_agent.solve(sentence_2, question_17))  # "mile"
#
# # Was Group
# print(test_agent.solve(sentence_2, question_10))  # "no"
# print(test_agent.solve(sentence_2, question_22))  # "Morning"
# print(test_agent.solve(sentence_2, question_23))  # "Yes"
# print(test_agent.solve(sentence_2, question_24))  # "No"
#
# # Did Group
# print(test_agent.solve(sentence_2, question_11))  # "Morning"
# print(test_agent.solve(sentence_2, question_32))  # "Yes"
# print(test_agent.solve(sentence_2, question_33))  # "No"
# print(test_agent.solve(sentence_2, question_34))  # "No"


if __name__ == "__main__":
	test()
	exit()

