import re


class QuestionFrame:
	# Person, place or thing
	def __init__(self, question_type=None):
		self.question_type = question_type
		pass


class SentenceReadingAgent:
	def __init__(self):
		# If you want to do any initial processing, add it here.
		self.most_common_pos = {}
		self.pos_adv_lookup = {}
		self.question_pos = None
		self.question = None
		self.sentence_pos = None
		self.sentence = None
		# https://stackoverflow.com/questions/33906033/regex-for-time-in-hhmm-am-pm-format
		self.time_regex = re.compile("(\d{1,2}\:\d{2}\s?(?:AM|PM|am|pm))")
		# self.setup()
		self.load_pos()
		pass
	
	def setup(self):
		# import spacy
		# nlp = spacy.load("en_core_web_trf")
		# with open("mostcommon.txt", "r") as input_file:
		#     t = input_file.read().splitlines()
		#     for i in t:
		#         doc = nlp(i)
		#         for token in doc:
		#             self.most_common_pos[token.doc.text] = {"Lemma": token.lemma_, "POS": token.pos_, "POS_adv": token.tag_}
		#             if token.pos_ not in self.pos_lookup:
		#                 self.pos_lookup[token.pos_] = spacy.explain(f"{token.pos_}")
		#             if token.tag_ not in self.pos_adv_lookup:
		#                 self.pos_adv_lookup[token.tag_] = spacy.explain(f"{token.tag_}")
		#     input_file.close()
		
		# with open("pos_lookup.py", "wb") as output_file:
		#     pickle.dump(self.pos_lookup, output_file)
		#     output_file.close()
		#
		# with open("pos_adv_lookup.py", "wb") as output_file:
		#     pickle.dump(self.pos_adv_lookup, output_file)
		#     output_file.close()
		#
		# with open("mostcommon_POS.py", "wb") as output_file:
		#     pickle.dump(self.most_common_pos, output_file)
		#     output_file.close()
		
		# with open("pos_lookup.py", "rb") as input_file:
		# 	self.pos_lookup = pickle.load(input_file)
		# 	input_file.close()
		
		# with open("pos_adv_lookup.py", "rb") as input_file:
		# 	self.pos_adv_lookup = pickle.load(input_file)
		# 	input_file.close()
		#
		# with open("mostcommon_POS.py", "rb") as input_file:
		# 	self.most_common_pos = pickle.load(input_file)
		# 	input_file.close()
		#
		return
	
	def find_time(self, sentence):
		results = re.findall(self.time_regex, sentence)
		return results
	
	def solve(self, sentence, question):
		# Add your code here! Your solve method should receive
		# two strings as input: sentence and question. It should
		# return a string representing the answer to the question.
		self.sentence = sentence
		self.question = question

		if "time" in question.lower():
			regex_results = re.findall(self.time_regex, sentence)
			if len(regex_results) > 0:
				return regex_results[0]
		
		sentence_split = sentence.split(" ")
		sentence_split[-1] = sentence_split[-1].replace(".", "")
		
		question_split = question.split(" ")
		question_split[-1] = question_split[-1].replace("?", "")
		
		result = self.run_filters(sentence_pos=sentence_split, question_pos=question_split)
		
		return result
	
	def run_filters(self, sentence_pos, question_pos):
		
		if question_pos[0].lower() == "who":
			return self.who_filter(sentence_pos=sentence_pos, question_pos=question_pos)
		elif question_pos[0].lower() == "what":
			return self.what_filter(sentence_pos=sentence_pos, question_pos=question_pos)
		elif question_pos[0].lower() == "when":
			return self.when_filter(sentence_pos=sentence_pos, question_pos=question_pos)
		elif question_pos[0].lower() == "where":
			return self.where_filter(sentence_pos=sentence_pos, question_pos=question_pos)
		elif question_pos[0].lower() == "why":
			return self.why_filter(sentence_pos=sentence_pos, question_pos=question_pos)
		elif question_pos[0].lower() == "how":
			return self.how_filter(sentence_pos=sentence_pos, question_pos=question_pos)
		elif question_pos[0].lower() == "was":
			return self.was_filter(sentence_pos=sentence_pos, question_pos=question_pos)
		elif question_pos[0].lower() == "did":
			return self.did_filter(sentence_pos=sentence_pos, question_pos=question_pos)
		else:
			if len(question_pos) > 0:
				del (question_pos[0])
				return self.run_filters(sentence_pos=sentence_pos, question_pos=question_pos)
	
	def make_bucket(self, some_sentence):
		bucket = {}
		for i in some_sentence:
			if i.lower() in self.most_common_pos:
				if self.most_common_pos[i.lower()]["POS"] == "VERB":
					if "VERB" not in bucket:
						bucket["VERB"] = []
					bucket["VERB"].append(i)
				elif self.most_common_pos[i.lower()]["POS"] == "NOUN":
					if "NOUN" not in bucket:
						bucket["NOUN"] = []
					bucket["NOUN"].append(i)
				elif self.most_common_pos[i.lower()]["POS"] == "PRONOUN":
					if "PRONOUN" not in bucket:
						bucket["PRONOUN"] = []
					bucket["PRONOUN"].append(i)
				elif self.most_common_pos[i.lower()]["POS"] == "CONJUNCTION":
					if "CONJUNCTION" not in bucket:
						bucket["CONJUNCTION"] = []
					bucket["CONJUNCTION"].append(i)
				elif self.most_common_pos[i.lower()]["POS"] == "NUMBER":
					if "NUMBER" not in bucket:
						bucket["NUMBER"] = []
					bucket["NUMBER"].append(i)
				elif self.most_common_pos[i.lower()]["POS"] == "PREPOSITION":
					if "PREPOSITION" not in bucket:
						bucket["PREPOSITION"] = []
					bucket["PREPOSITION"].append(i)
				elif self.most_common_pos[i.lower()]["POS"] == "ADVERB":
					if "ADVERB" not in bucket:
						bucket["ADVERB"] = []
					bucket["ADVERB"].append(i)
				elif self.most_common_pos[i.lower()]["POS"] == "ADJECTIVE":
					if "ADJECTIVE" not in bucket:
						bucket["ADJECTIVE"] = []
					bucket["ADJECTIVE"].append(i)
				elif self.most_common_pos[i.lower()]["POS"] == "TIME":
					if "TIME" not in bucket:
						bucket["TIME"] = []
					bucket["TIME"].append(i)
				elif self.most_common_pos[i.lower()]["POS"] == "OTHER":
					if self.most_common_pos[i.lower()]["POS_adv"] == "DT":
						if "DETERMINER" not in bucket:
							bucket["DETERMINER"] = []
						bucket["DETERMINER"].append(i)
					else:
						if "OTHER" not in bucket:
							bucket["OTHER"] = []
						bucket["OTHER"].append(i)
				elif self.most_common_pos[i.lower()]["POS"] == "REMOVE":
					if "REMOVE" not in bucket:
						bucket["REMOVE"] = []
					bucket["REMOVE"].append(i)
		return bucket
	
	def who_filter(self, sentence_pos, question_pos):
		possible_answers = []
		possible_answers_pos = []
		for i in sentence_pos:
			if i in self.most_common_pos:
				if self.most_common_pos[i]["Category"] in ("PERSON", "NAME", "PRONOUN"):
					possible_answers.append(i)
					possible_answers_pos.append(self.most_common_pos[i])
			elif i.lower() in self.most_common_pos:
				if self.most_common_pos[i.lower()]["Category"] in ("PERSON", "NAME", "PRONOUN"):
					possible_answers.append(i.lower())
					possible_answers_pos.append(self.most_common_pos[i.lower()])
		
		if len(possible_answers) == 1:
			return possible_answers[0]
		else:
			question_pos.remove("Who")
			question_parts_person = []
			question_parts_person_pos = []
			question_parts_action = []
			question_parts_action_pos = []
			for i in question_pos:
				if i in self.most_common_pos:
					if self.most_common_pos[i]["Category"] in ("PERSON", "NAME", "PRONOUN"):
						if i not in question_parts_person:
							question_parts_person.append(i)
							question_parts_person_pos.append(self.most_common_pos[i])
					elif self.most_common_pos[i]["Category"] in ("ACTION"):
						if i not in question_parts_action:
							question_parts_action.append(i)
							question_parts_action_pos.append(self.most_common_pos[i])
				else:
					if i.lower() in self.most_common_pos:
						if self.most_common_pos[i.lower()]["Category"] in ("PERSON", "NAME", "PRONOUN"):
							if i in question_parts_person:
								question_parts_person.append(i)
								question_parts_person_pos.append(self.most_common_pos[i.lower()])
						elif self.most_common_pos[i.lower()]["Category"] in ("ACTION"):
							if i in question_parts_action:
								question_parts_action.append(i)
								question_parts_action_pos.append(self.most_common_pos[i.lower()])
			
			for i in question_parts_person:
				if i in possible_answers:
					possible_answers.remove(i)
					if len(possible_answers) == 1:
						return possible_answers[0]
			
			if len(question_parts_action) > 0:
				idx = 0
				idx_set = False
				q_part = question_parts_action[0]
				if q_part.lower() in self.most_common_pos:
					q_lemma = self.most_common_pos[q_part.lower()]["Lemma"]
					for j in question_parts_action:
						if j in sentence_pos:
							for i in range(len(sentence_pos)):
								if sentence_pos[i] in self.most_common_pos:
									temp = self.most_common_pos[sentence_pos[i]]["Lemma"]
								else:
									temp = self.most_common_pos[sentence_pos[i].lower()]["Lemma"]
								if q_lemma == temp:
									idx = i
									idx_set = True
				new_possible_answers = []
				if "is" in question_pos:
					for i in sentence_pos:
						if i not in question_pos:
							if i in self.most_common_pos:
								if self.most_common_pos[i]["POS"] != "DET":
									new_possible_answers.append(i)
				if len(new_possible_answers) > 0:
					if new_possible_answers[-1] in self.most_common_pos:
						if self.most_common_pos[new_possible_answers[-1]]["Category"] == "PERSON":
							return new_possible_answers[-1]
							
				temp_results = []
				for i in sentence_pos[:idx]:
					if i in self.most_common_pos:
						if self.most_common_pos[i]["Category"] in ("PERSON", "NAME", "PRONOUN"):
							temp_results.append(i)
					elif i.lower() in self.most_common_pos:
						if self.most_common_pos[i.lower()]["Category"] in ("PERSON", "NAME", "PRONOUN"):
							temp_results.append(i.lower())
				
				if len(temp_results) == 1:
					return temp_results[0]
				
				if len(question_parts_action) > 0:
					if idx_set:
						temp_results = sentence_pos[:idx]
						if len(temp_results) == 1:
							return temp_results[0]
						else:
							for ii in range(len(temp_results)):
								if temp_results[ii] in self.most_common_pos:
									t = self.most_common_pos[temp_results[ii]]
									if t["Category"] in ("PERSON", "NAME", "PRONOUN"):
										return temp_results[ii]
		if len(question_parts_action) > 0:
			for i in question_parts_action:
				if i.lower() in self.most_common_pos:
					if self.most_common_pos[i.lower()]["Category"] != "ACTION":
						question_parts_action.remove(i)
			if len(question_parts_action) == 1:
				same_lemma_idx = 0
				same_lemma_set = False
				for i in sentence_pos:
					if i.lower() in self.most_common_pos:
						if self.most_common_pos[i.lower()]["Lemma"] == self.most_common_pos[question_parts_action[0]]["Lemma"]:
							same_lemma_idx = sentence_pos.index(i)
							same_lemma_set = True
				if same_lemma_set:
					if "was" in question_pos:
						for i in range(same_lemma_idx - 1, -1, -1):
							if self.most_common_pos[sentence_pos[i].lower()]["Category"] in ("PRONOUN", "NAME"):
								if sentence_pos[i].lower() in possible_answers:
									possible_answers.remove(sentence_pos[i].lower())
					else:
						for i in range(same_lemma_idx-1, -1, -1):
							if self.most_common_pos[sentence_pos[i].lower()]["Category"] in ("PRONOUN", "NAME"):
								return sentence_pos[i]
		
		if len(possible_answers) == 1:
			return possible_answers[0]
		elif len(possible_answers) == 2:
			return possible_answers[0] + " " + possible_answers[1]
				
		return
	
	def was_filter(self, sentence_pos, question_pos):
		if "or" in question_pos:
			or_idx = question_pos.index("or")
			possible_answers = [question_pos[or_idx - 1], question_pos[or_idx + 1]]
			action_idx = 0
			for i in range(len(question_pos) - 1, 0, -1):
				if question_pos[i] in self.most_common_pos:
					if self.most_common_pos[question_pos[i]]["Category"] == "ACTION":
						if "AM " in self.sentence or "am " in self.sentence:
							if "noon" in possible_answers[0]:
								return possible_answers[1]
							else:
								return possible_answers[0]
						else:
							if "noon" in possible_answers[0]:
								return possible_answers[0]
							else:
								return possible_answers[1]
		
		elif "in" in question_pos:
			# Answers are Yes or No
			in_idx = question_pos.index("in")
			if self.most_common_pos[question_pos[in_idx + 1]]["POS"] == "DET":
				value_check_if_true = question_pos[in_idx + 2]
			else:
				value_check_if_true = question_pos[in_idx + 1]
			if "noon" in value_check_if_true:
				if value_check_if_true in self.sentence:
					return "Yes"
				else:
					if "PM " in self.sentence or "pm " in self.sentence:
						return "Yes"
					return "No"
		if "morning" in question_pos:
			if "am " in self.sentence or "AM " in self.sentence:
				return "Yes"
			else:
				return "No"
		found_idx = 0
		found_set = False
		for i in range(len(sentence_pos) - 1, 0, -1):
			if self.most_common_pos[sentence_pos[i]]["POS"] == "AUX":
				found_set = True
				found_idx = i
				break
		if found_set:
			return sentence_pos[found_idx + 1]
		return "Yes"
	
	def did_filter(self, sentence_pos, question_pos):
		if "or" in question_pos:
			or_idx = question_pos.index("or")
			possible_answers = [question_pos[or_idx - 1], question_pos[or_idx + 1]]
			if possible_answers[0] in sentence_pos:
				return possible_answers[0]
			elif possible_answers[1] in sentence_pos:
				return possible_answers[1]
		
		return None
	
	def what_filter(self, sentence_pos, question_pos):
		possible_answers = []
		possible_answers_pos = []
		look_for_action = False
		new_possible_answers = None
		if "color" in question_pos or "Color" in question_pos:
			# Find person/place/thing after 'is'
			agent_idx = 0
			agent_idx_set = False
			if "is" in question_pos or "was" in question_pos:
				if "is" in question_pos:
					is_idx = question_pos.index("is")
				else:
					is_idx = question_pos.index("was")
				for i in range(len(question_pos[is_idx:])):
					actual_idx = is_idx + i
					if question_pos[actual_idx] in self.most_common_pos:
						if self.most_common_pos[question_pos[actual_idx]]["Category"] in ("THING", "MULTI", "OBJECT"):
							agent_idx = actual_idx
							agent_idx_set = True
					elif question_pos[actual_idx].lower() in self.most_common_pos:
						if self.most_common_pos[question_pos[actual_idx].lower()]["Category"] in (
						"THING", "MULTI", "OBJECT"):
							agent_idx = actual_idx
							agent_idx_set = True
			if agent_idx_set:
				if question_pos[agent_idx] in sentence_pos:
					object_idx = sentence_pos.index(question_pos[agent_idx])
				else:
					object_idx = sentence_pos.index(question_pos[agent_idx].lower())
				for i in range(object_idx, 0, -1):
					if sentence_pos[i] in self.most_common_pos:
						if self.most_common_pos[sentence_pos[i]]["Category"] == "COLOR":
							return sentence_pos[i]
					elif sentence_pos[i].lower() in self.most_common_pos:
						if self.most_common_pos[sentence_pos[i].lower()]["Category"] == "COLOR":
							return sentence_pos[i]
				possible_answers = []
				for i in range(len(sentence_pos)):
					if sentence_pos[i] in self.most_common_pos:
						if self.most_common_pos[sentence_pos[i]]["Category"] == "COLOR":
							possible_answers.append(sentence_pos[i])
				if len(possible_answers) == 1:
					return possible_answers[0]
				closest = 100
				closest_idx = 100
				for i in range(len(possible_answers)):
					diff = abs(sentence_pos.index(possible_answers[i]) - object_idx)
					if diff < closest:
						closest = diff
						closest_idx = i
				if closest_idx != 100:
					return possible_answers[closest_idx]
		
		if question_pos[-1] in ('do', 'did', 'does'):
			look_for_action = True
		if look_for_action:
			for i in sentence_pos:
				if i in self.most_common_pos:
					if self.most_common_pos[i]["Category"] in ("ACTION"):
						possible_answers.append(i)
						possible_answers_pos.append(self.most_common_pos[i])
				elif i.lower() in self.most_common_pos:
					if self.most_common_pos[i.lower()]["Category"] in ("ACTION"):
						possible_answers.append(i.lower())
						possible_answers_pos.append(self.most_common_pos[i.lower()])
			if len(possible_answers) == 1:
				return possible_answers[0]
			
			actor_idx = 0
			actor_idx_set = False
			for i in range(len(question_pos)):
				if question_pos[i] in self.most_common_pos:
					if self.most_common_pos[question_pos[i]]["Category"] in ("NAME", "PERSON", "PRONOUN"):
						actor_idx = i
						actor_idx_set = True
				elif question_pos[i].lower() in self.most_common_pos:
					if self.most_common_pos[question_pos[i].lower()]["Category"] in ("NAME", "PERSON", "PRONOUN"):
						actor_idx = i
						actor_idx_set = True
			if actor_idx_set:
				sentence_actor_idx = sentence_pos.index(question_pos[actor_idx])
				nxt_idx = 100
				best_diff = 100
				for i in range(len(possible_answers)):
					ans_idx = sentence_pos.index(possible_answers[i])
					diff = abs(sentence_actor_idx - ans_idx)
					if diff < best_diff:
						best_diff = diff
						nxt_idx = i
				if nxt_idx != 100:
					return possible_answers[nxt_idx]
		else:
			if self.most_common_pos[question_pos[-1]]["Category"] == "ACTION":
				if question_pos[-1] in sentence_pos:
					action_idx_in_question = sentence_pos.index(question_pos[-1])
					for i in range(len(sentence_pos[action_idx_in_question:])):
						actual_idx = action_idx_in_question + i
						if sentence_pos[actual_idx] != sentence_pos[action_idx_in_question]:
							if self.most_common_pos[sentence_pos[actual_idx]]["POS"] in ("VERB", "NOUN", "PROPN"):
								return sentence_pos[actual_idx]
							elif self.most_common_pos[sentence_pos[actual_idx]]["POS"] in ("DET"):
								return sentence_pos[actual_idx + 1]
			
			if "will" in question_pos:
				# first_set_possible_answers = [i for i in sentence_pos if
				#                               i not in question_pos or i.lower() not in question_pos and i.lower() in self.most_common_pos and
				#                               self.most_common_pos[i.lower()]["POS"] != "DET"]
				first_set_possible_answers = []
				for i in sentence_pos:
					if i in question_pos or i.lower() in question_pos:
						continue
					else:
						first_set_possible_answers.append(i)
				for val in first_set_possible_answers:
					if val in self.most_common_pos:
						if self.most_common_pos[val]["Category"] == "INTERJECTION":
							first_set_possible_answers.remove(val)

				if len(first_set_possible_answers) == 2:
					noun_count = 0
					adj_count = 0
					adj_idx = 0
					noun_idx = 0
					noun_flag = False
					adj_flag = False
					multi_count = 0
					multi_idx = 0
					multi_flag = False
					for i in first_set_possible_answers:
						if i.lower() in self.most_common_pos:
							if self.most_common_pos[i.lower()]["POS"] == "ADJ":
								adj_count += 1
								adj_idx = first_set_possible_answers.index(i)
								adj_flag = True
							elif self.most_common_pos[i.lower()]["POS"] in ("PROPN", "NOUN") and self.most_common_pos[i.lower()]["Category"] != "MULTI":
								noun_count += 1
								noun_idx = first_set_possible_answers.index(i)
								noun_flag = True
							elif self.most_common_pos[i.lower()]["Category"] == "MULTI":
								multi_count += 1
								multi_idx = first_set_possible_answers.index(i)
								multi_flag = True
					if noun_count == 1 and adj_count == 1 and noun_flag and adj_flag:
						return first_set_possible_answers[adj_idx] + " " + first_set_possible_answers[noun_idx]
					elif noun_count == 1 and multi_count == 1 and multi_flag and noun_flag:
						return first_set_possible_answers[multi_idx] + " " + first_set_possible_answers[noun_idx]
					elif noun_count >= 1 and noun_flag:
						return first_set_possible_answers[noun_idx]
			
			for i in sentence_pos:
				if i in self.most_common_pos:
					if self.most_common_pos[i]["Category"] in ("OBJECT", "ACTION"):
						if self.most_common_pos[i]["Lemma"] != self.most_common_pos[question_pos[-1]]["Lemma"] \
								and self.most_common_pos[i]["Lemma"] != self.most_common_pos[question_pos[0].lower()][
							"Lemma"]:
							possible_answers.append(i)
							possible_answers_pos.append(self.most_common_pos[i])
				elif i.lower() in self.most_common_pos:
					if self.most_common_pos[i.lower()]["Category"] in ("OBJECT", "ACTION"):
						if self.most_common_pos[i.lower()]["Lemma"] != self.most_common_pos[question_pos[-1]]["Lemma"] \
								and self.most_common_pos[i.lower()]["Lemma"] != \
								self.most_common_pos[question_pos[0].lower()]["Lemma"]:
							possible_answers.append(i.lower())
							possible_answers_pos.append(self.most_common_pos[i.lower()])
			if len(possible_answers) == 1:
				if possible_answers[0] in self.most_common_pos:
					if self.most_common_pos[possible_answers[0]]["Category"] != "ACTION":
						return possible_answers[0]
				is_answer = True
				same_lemma_idx = 100
				for i in sentence_pos:
					if i.lower() in self.most_common_pos:
						if self.most_common_pos[i.lower()]["Lemma"] == self.most_common_pos[possible_answers[0]][
							"Lemma"]:
							is_answer = False
							same_lemma_idx = sentence_pos.index(i)
							break
				if is_answer:
					return possible_answers[0]
				if same_lemma_idx != 100 and is_answer:
					if self.most_common_pos[sentence_pos[same_lemma_idx + 1]]["POS"] == "DET" or \
							self.most_common_pos[sentence_pos[same_lemma_idx + 1]]["POS"] == "INTJ":
						return sentence_pos[same_lemma_idx + 1] + " " + sentence_pos[same_lemma_idx + 2]
					else:
						return sentence_pos[same_lemma_idx + 1]
			else:
				new_possible_answers = []
				for i in possible_answers:
					if i in sentence_pos and i in question_pos:
						# possible_answers.remove(i)
						continue
					else:
						new_possible_answers.append(i)
				if len(possible_answers) == 1:
					return possible_answers[0]
				elif len(new_possible_answers) == 1:
					return new_possible_answers[0]
					
		if 'name' in question_pos:
			possible_answers = []
			for i in range(len(sentence_pos)):
				if sentence_pos[i] in self.most_common_pos:
					if self.most_common_pos[sentence_pos[i]]["Category"] == "Name":
						possible_answers.append(sentence_pos[i])
					elif i != 0 and sentence_pos[i][0].isupper():
						possible_answers.append(sentence_pos[i])
				elif sentence_pos[i].lower() in self.most_common_pos:
					if self.most_common_pos[sentence_pos[i].lower()]["Category"] == "Name":
						possible_answers.append(sentence_pos[i])
					elif i != 0 and sentence_pos[i][0].isupper():
						possible_answers.append(sentence_pos[i])
			if len(possible_answers) == 1:
				return possible_answers[0]
		if "is" in question_pos:
			if "is" in sentence_pos:
				if sentence_pos[sentence_pos.index("is") - 1] in self.most_common_pos:
					if self.most_common_pos[sentence_pos[sentence_pos.index("is") - 1]]["POS"] != "ADV":
						for i in range(sentence_pos.index("is") - 1, -1, -1):
							if sentence_pos[i] not in question_pos:
								return sentence_pos[i]
					else:
						return sentence_pos[sentence_pos.index("is") + 1]
			else:
				is_idx_in_question = question_pos.index("is")
				subject_in_question = is_idx_in_question + 1
				if question_pos[subject_in_question] in sentence_pos:
					subject_in_sentence_idx = sentence_pos.index(question_pos[subject_in_question])
					if sentence_pos[subject_in_sentence_idx].lower() in self.most_common_pos:
						if self.most_common_pos[sentence_pos[subject_in_sentence_idx - 1].lower()]["POS"] == "DET":
							return sentence_pos[subject_in_sentence_idx + 1]
						else:
							return sentence_pos[subject_in_sentence_idx - 1]
		actions = []
		for i in question_pos:
			if i.lower() in self.most_common_pos:
				if self.most_common_pos[i.lower()]["Category"] == "ACTION":
					actions.append(i.lower())
					
		action_idx_sentence_idx = 0
		action_idx_set = False
		if len(actions) > 0:
			for action in actions:
				if action in sentence_pos and self.most_common_pos[action]["Lemma"] == self.most_common_pos[sentence_pos[sentence_pos.index(action)]]["Lemma"]:
					action_idx_sentence_idx = sentence_pos.index(action)
					action_idx_set = True
				else:
					for i in sentence_pos:
						if i.lower() in self.most_common_pos:
							if self.most_common_pos[i.lower()]["Lemma"] == self.most_common_pos[action]["Lemma"]:
								action_idx_set = True
								action_idx_sentence_idx = sentence_pos.index(i)
		if action_idx_set:
			if sentence_pos[action_idx_sentence_idx + 1] in self.most_common_pos:
				if self.most_common_pos[sentence_pos[action_idx_sentence_idx + 1]]["POS"] == "DET":
					return sentence_pos[action_idx_sentence_idx + 1] + " " + sentence_pos[action_idx_sentence_idx + 2]
				else:
					sentence = ""
					for i in sentence_pos[action_idx_sentence_idx+1:]:
						if i.lower() in self.most_common_pos:
							if self.most_common_pos[i.lower()]["Category"] in ("MULTI", "NOUN", "INTERJECTION"):
								sentence += i + " "
					if len(sentence) > 0:
						sentence = sentence.rstrip()
						return sentence
					else:
						alternate_answers = []
						if new_possible_answers is not None:
							for i in new_possible_answers:
								if i not in actions:
									alternate_answers.append(i)
						
						if len(alternate_answers) == 1:
							return alternate_answers[-1]
						else:
							even_newer_possible_answers = []
							start_idx = 0
							start_idx_set = False
							end_idx = 0
							end_idx_set = False
							if new_possible_answers is not None:
								for i in new_possible_answers:
									for j in sentence_pos:
										if self.most_common_pos[i]["Lemma"] == self.most_common_pos[j.lower()]["Lemma"]:
											if not start_idx_set:
												start_idx = sentence_pos.index(j)
												start_idx_set = True
											else:
												end_idx = sentence_pos.index(j)
												end_idx_set = True
								if start_idx_set and end_idx_set:
									even_newer_possible_answers = sentence_pos[start_idx+1:end_idx]
									choices = []
									for i in even_newer_possible_answers:
										if i in self.most_common_pos:
											if self.most_common_pos[i]["POS"] not in ("PRON", "NOUN"):
												choices.append(i)
									if len(choices) == 1:
										return choices[-1]
		if len(actions) > 1:
			action_idx_in_question = question_pos.index(actions[-1])
			action_in_question = question_pos[action_idx_in_question]
			action_idx_in_sentence = None
			action_in_sentence = None

			for i in sentence_pos:
				if i.lower() in self.most_common_pos:
					if self.most_common_pos[i.lower()]["Lemma"] == self.most_common_pos[action_in_question]["Lemma"]:
						action_idx_in_sentence = sentence_pos.index(i)
						action_in_sentence = sentence_pos[action_idx_in_sentence]
						break
			if action_idx_in_sentence is not None:
				new_sentence_pos = sentence_pos[action_idx_in_sentence+1:]
				for i in question_pos:
					if i.lower() in new_sentence_pos:
						new_sentence_pos.remove(i.lower())
		if "was" in question_pos:
			was_idx = question_pos.index('was')
			remaining = question_pos[was_idx+1:]
			if len(remaining) == 1:
				if remaining[-1] in sentence_pos:
					target_idx = sentence_pos.index(remaining[-1])
					if sentence_pos[target_idx+1] in self.most_common_pos:
						if self.most_common_pos[sentence_pos[target_idx+1]]["POS"] == "NOUN" or self.most_common_pos[sentence_pos[target_idx+1]]["Category"] == "MULTI":
							return sentence_pos[target_idx+1]
		return None
	
	def when_filter(self, sentence_pos, question_pos):
		possible_answers = []
		possible_answers_pos = []
		for i in sentence_pos:
			if i in self.most_common_pos:
				if self.most_common_pos[i]["Category"] in ("TIME", "DETERMINER"):
					possible_answers.append(i)
					possible_answers_pos.append(self.most_common_pos[i])
			elif i.lower() in self.most_common_pos:
				if self.most_common_pos[i.lower()]["Category"] in ("TIME", "DETERMINER"):
					possible_answers.append(i.lower())
					possible_answers_pos.append(self.most_common_pos[i.lower()])
		
		if len(possible_answers) == 1:
			return possible_answers[0]
		else:
			question_parts = []
			question_parts_pos = []
			for i in question_pos:
				if i in self.most_common_pos:
					if self.most_common_pos[i]["Category"] in ("TIME", "DETERMINER"):
						question_parts.append(i)
						question_parts_pos.append(self.most_common_pos[i])
				elif i.lower() in self.most_common_pos:
					if self.most_common_pos[i.lower()]["Category"] in ("TIME", "DETERMINER"):
						question_parts.append(i.lower())
						question_parts_pos.append(self.most_common_pos[i.lower()])
			
			for i in question_parts:
				if i in possible_answers:
					possible_answers.remove(i)
					if len(possible_answers) == 1:
						return possible_answers[0]
			if len(possible_answers) == 2:
				det_count = 0
				det_idx = 0
				time_count = 0
				time_idx = 0
				for i in range(len(possible_answers)):
					if self.most_common_pos[possible_answers[i]]["Category"] == "TIME":
						time_count += 1
						time_idx = i
					elif self.most_common_pos[possible_answers[i]]["Category"] == "DETERMINER":
						det_count += 1
						det_idx = i
				if det_count == 1 and time_count == 1:
					result = possible_answers[det_idx] + " " + possible_answers[time_idx]
					return result
		# This point no valid answers
		regex_results = re.findall(self.time_regex, self.sentence)
		if len(regex_results) > 0:
			return regex_results[0]
		if question_pos[-1] in sentence_pos:
			return sentence_pos[sentence_pos.index(question_pos[-1]) + 1]
		return
	
	def where_filter(self, sentence_pos, question_pos):
		possible_answers = []
		possible_answers_pos = []
		for i in sentence_pos:
			if i in self.most_common_pos:
				if self.most_common_pos[i]["Category"] in ("PLACE", "MULTI"):
					possible_answers.append(i)
					possible_answers_pos.append(self.most_common_pos[i])
			elif i.lower() in self.most_common_pos:
				if self.most_common_pos[i.lower()]["Category"] in ("PLACE", "MULTI"):
					possible_answers.append(i.lower())
					possible_answers_pos.append(self.most_common_pos[i.lower()])
		
		if len(possible_answers) == 1:
			return possible_answers[0]
		else:
			
			question_parts = []
			question_parts_pos = []
			for i in question_pos:
				if i in self.most_common_pos:
					if self.most_common_pos[i]["Category"] in ("PLACE", "MULTI"):
						question_parts.append(i)
						question_parts_pos.append(self.most_common_pos[i])
				elif i.lower() in self.most_common_pos:
					if self.most_common_pos[i.lower()]["Category"] in ("PLACE", "MULTI"):
						question_parts.append(i.lower())
						question_parts_pos.append(self.most_common_pos[i.lower()])
			
			for i in question_parts:
				if i in possible_answers:
					possible_answers.remove(i)
					if "is" in question_pos and "is" in sentence_pos:
						is_idx = sentence_pos.index("is")
						noun_idx = None
						if len(possible_answers) == 1:
							noun_idx = sentence_pos.index(possible_answers[0])
						if noun_idx is not None:
							sentence = ""
							for i in sentence_pos[is_idx+1:noun_idx+1]:
								sentence += i + " "
							sentence = sentence.rstrip()
							return sentence
						else:
							return possible_answers[0]
		if len(possible_answers) == 1:
			return possible_answers[0]
		return
	
	def why_filter(self, sentence_pos, question_pos):
		
		return
	
	def how_filter(self, sentence_pos, question_pos):
		question_pos.remove("How")
		if "much" in question_pos:
			possible_answers = []
			possible_answers_pos = []
			for i in sentence_pos:
				if i in self.most_common_pos:
					if self.most_common_pos[i]["POS"] in ("DET"):
						possible_answers.append(i)
						possible_answers_pos.append(self.most_common_pos[i])
				elif i.lower() in self.most_common_pos:
					if self.most_common_pos[i.lower()]["POS"] in ("DET"):
						possible_answers.append(i.lower())
						possible_answers_pos.append(self.most_common_pos[i.lower()])
			if len(possible_answers) == 1:
				return possible_answers[0]
		elif "many" in question_pos:
			possible_answers = []
			possible_answers_pos = []
			for i in sentence_pos:
				if i in self.most_common_pos:
					if self.most_common_pos[i]["Category"] == "NUMBER":
						possible_answers.append(i)
						possible_answers_pos.append(self.most_common_pos[i])
				elif i.lower() in self.most_common_pos:
					if self.most_common_pos[i.lower()]["Category"] == "NUMBER":
						possible_answers.append(i.lower())
						possible_answers_pos.append(self.most_common_pos[i.lower()])
			if len(possible_answers) == 1:
				return possible_answers[0]
			if len(possible_answers) == 2:
				idx_1 = sentence_pos.index(possible_answers[0])
				idx_2 = sentence_pos.index(possible_answers[1])
				if abs(idx_1 - idx_2) == 1:
					sentence = possible_answers[0] + " " + possible_answers[1]
					return sentence
		start_val = question_pos[0]
		start_val_pos = self.most_common_pos[start_val]
		if start_val_pos["POS"] == "ADJ":
			# describe noun
			# Find Noun
			the_noun_idx = 0
			noun_idx_set = False
			for i in range(len(question_pos)):
				if question_pos[i] in self.most_common_pos:
					if self.most_common_pos[question_pos[i]]["Category"] in ("OBJECT", "MULTI"):
						the_noun_idx = i
						noun_idx_set = True
						break
			if noun_idx_set:
				if question_pos[the_noun_idx] in sentence_pos:
					return sentence_pos[sentence_pos.index(question_pos[the_noun_idx]) - 1]
		elif start_val_pos["POS"] == "ADV":
			# Descrive verb
			# Find verb
			the_verb_idx = 0
			verb_idx_set = False
			for i in range(len(question_pos) - 1, 0, -1):
				if question_pos[i] in self.most_common_pos:
					if self.most_common_pos[question_pos[i]]["Category"] == "ACTION":
						the_verb_idx = i
						verb_idx_set = True
						break
			verb = question_pos[the_verb_idx]
			if verb in sentence_pos:
				verb_idx_in_string = sentence_pos.index(verb)
				word = sentence_pos[verb_idx_in_string + 1]
				if self.most_common_pos[word]["Category"] == "NUMBER":
					return word + " " + sentence_pos[verb_idx_in_string + 2]
				else:
					return word
		idx = sentence_pos.index(question_pos[-1])
		for i in sentence_pos[:idx]:
			if i in self.most_common_pos:
				if self.most_common_pos[i]["Category"] == "ACTION":
					return i
		if "is" in question_pos and "is" in sentence_pos:
			is_idx = sentence_pos.index("is")
			if sentence_pos[is_idx+1].lower() in self.most_common_pos:
				if self.most_common_pos[sentence_pos[is_idx+1].lower()]["Category"] == "DESC VERB":
					return sentence_pos[is_idx+1] + " " + sentence_pos[is_idx + 2]
				else:
					return sentence_pos[is_idx + 1]
		return
	
	def load_pos(self):
		# with open('dict.csv', 'w') as csv_file:
		# 	for key, value in self.most_common_pos.items():
		# 		temp_string = f"'{key}': " "{" + f"'Lemma': '{value['Lemma']}', 'POS': '{value['POS']}', 'POS_adv': '{value['POS_adv']}', 'Category': 'ACTION'" + "}"
		# 		csv_file.write(temp_string + ",\n")
		# resu = {}
		# for key, val in self.most_common_pos.items():
		# 	if self.most_common_pos[key]["POS"] not in resu:
		# 		resu[self.most_common_pos[key]["POS"]] = {}
		# 	resu[self.most_common_pos[key]["POS"]][key] = val
		# t = {k: v for k, v in sorted(resu["PROPN"].items(), key=lambda ky: ky)}
		# with open('dict.csv', 'w') as csv_file:
		# 	for key, value in resu.items():
		# 		t = {k: v for k, v in sorted(value.items(), key=lambda ky: ky)}
		# 		for kk, vv in t.items():
		# 			tes_str = ""
		# 			if vv['POS'] == "NOUN":
		# 				tes_str = "OBJECT"
		# 			elif vv['POS'] == "PROPN":
		# 				tes_str = "NAME"
		# 			elif vv['POS'] == "ADJ":
		# 				tes_str = "DESC NOUN"
		# 			elif vv['POS'] == "ADV":
		# 				tes_str = "DESC VERB"
		# 			elif vv['POS'] == "VERB":
		# 				tes_str = "ACTION"
		# 			elif vv['POS'] == "NUM":
		# 				tes_str = "NUMBER"
		# 			elif vv['POS'] == "PRON":
		# 				tes_str = "PRONOUN"
		# 			elif vv['POS'] == "DET":
		# 				tes_str = "DETERMINER"
		# 			elif vv['POS'] == "CCONJ" or vv['POS_adv'] == "IN":
		# 				tes_str = "CONJUNCTION"
		# 			elif vv['POS'] == "INTJ":
		# 				tes_str = "INTERJECTION"
		# 			temp_string = f"'{kk}': " "{" + f"'Lemma': '{vv['Lemma']}', 'POS': '{vv['POS']}', " \
		# 			                                f"'POS_adv': '{vv['POS_adv']}', 'Category': '{tes_str}', " \
		# 			                                f"'POS_adv_desc': '{self.pos_adv_lookup[vv['POS_adv']]}'" + "}"
		# 			csv_file.write(temp_string + ",\n")
		self.most_common_pos = {'Ada': {'Lemma': 'Ada', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                'POS_adv_desc': 'noun, proper singular'},
		                        'Andrew': {'Lemma': 'Andrew', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                   'POS_adv_desc': 'noun, proper singular'},
		                        'Red': {'Lemma': 'Red', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                'POS_adv_desc': 'noun, proper singular'},
		                        'Bobbie': {'Lemma': 'Bobbie', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                   'POS_adv_desc': 'noun, proper singular'},
		                        'Cason': {'Lemma': 'Cason', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                  'POS_adv_desc': 'noun, proper singular'},
		                        'David': {'Lemma': 'David', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                  'POS_adv_desc': 'noun, proper singular'},
		                        'Farzana': {'Lemma': 'Farzana', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                    'POS_adv_desc': 'noun, proper singular'},
		                        'Frank': {'Lemma': 'Frank', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                  'POS_adv_desc': 'noun, proper singular'},
		                        'Hannah': {'Lemma': 'Hannah', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                   'POS_adv_desc': 'noun, proper singular'},
		                        'Ida': {'Lemma': 'Ida', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                'POS_adv_desc': 'noun, proper singular'},
		                        'Irene': {'Lemma': 'Irene', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                  'POS_adv_desc': 'noun, proper singular'},
		                        'Jim': {'Lemma': 'Jim', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                'POS_adv_desc': 'noun, proper singular'},
		                        'Jose': {'Lemma': 'Jose', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                 'POS_adv_desc': 'noun, proper singular'},
		                        'Keith': {'Lemma': 'Keith', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                  'POS_adv_desc': 'noun, proper singular'},
		                        'Laura': {'Lemma': 'Laura', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                  'POS_adv_desc': 'noun, proper singular'},
		                        'Lucy': {'Lemma': 'Lucy', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                 'POS_adv_desc': 'noun, proper singular'},
		                        'Meredith': {'Lemma': 'Meredith', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                     'POS_adv_desc': 'noun, proper singular'},
		                        'Nick': {'Lemma': 'Nick', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                 'POS_adv_desc': 'noun, proper singular'},
		                        'Serena': {'Lemma': 'Serena', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                   'POS_adv_desc': 'noun, proper singular'},
		                        'Yan': {'Lemma': 'Yan', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                'POS_adv_desc': 'noun, proper singular'},
		                        'Yeeling': {'Lemma': 'Yeeling', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                    'POS_adv_desc': 'noun, proper singular'},
		                        'ago': {'Lemma': 'ago', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'DESC NOUN',
		                                'POS_adv_desc': 'noun, proper singular'},
		                        'an': {'Lemma': 'an', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'DESC NOUN',
		                               'POS_adv_desc': 'noun, proper singular'},
		                        'can': {'Lemma': 'can', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'DESC NOUN',
		                                'POS_adv_desc': 'noun, proper singular'},
		                        'main': {'Lemma': 'main', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'noun, proper singular'},
		                        'ran': {'Lemma': 'ran', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NAME',
		                                'POS_adv_desc': 'noun, proper singular'},
		                        'snow': {'Lemma': 'snow', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'NOUN',
		                                 'POS_adv_desc': 'noun, proper singular'},
		                        'took': {'Lemma': 'took', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'VERB',
		                                 'POS_adv_desc': 'noun, proper singular'},
		                        'vowel': {'Lemma': 'vowel', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, proper singular'},
		                        'will': {'Lemma': 'will', 'POS': 'PROPN', 'POS_adv': 'NNP', 'Category': 'MULTI',
		                                 'POS_adv_desc': 'noun, proper singular'},
		                        'Adult': {'Lemma': 'adult', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'PERSON',
		                                  'POS_adv_desc': 'adjective'},
		                        'adult': {'Lemma': 'adult', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'PERSON',
		                                  'POS_adv_desc': 'adjective'},
		                        'Adults': {'Lemma': 'adult', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'PERSON',
		                                  'POS_adv_desc': 'adjective'},
		                        'adults': {'Lemma': 'adult', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'PERSON',
		                                  'POS_adv_desc': 'adjective'},
		                        'able': {'Lemma': 'able', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'big': {'Lemma': 'big', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                'POS_adv_desc': 'adjective'},
		                        'black': {'Lemma': 'black', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'COLOR',
		                                  'POS_adv_desc': 'adjective'},
		                        'blue': {'Lemma': 'blue', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'COLOR',
		                                 'POS_adv_desc': 'adjective'},
		                        'busy': {'Lemma': 'busy', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'certain': {'Lemma': 'certain', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                    'POS_adv_desc': 'adjective'},
		                        'clear': {'Lemma': 'clear', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                  'POS_adv_desc': 'adjective'},
		                        'cold': {'Lemma': 'cold', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'common': {'Lemma': 'common', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                   'POS_adv_desc': 'adjective'},
		                        'complete': {'Lemma': 'complete', 'POS': 'ADJ', 'POS_adv': 'JJ',
		                                     'Category': 'DESC NOUN', 'POS_adv_desc': 'adjective'},
		                        'cool': {'Lemma': 'cool', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'correct': {'Lemma': 'correct', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                    'POS_adv_desc': 'adjective'},
		                        'dark': {'Lemma': 'dark', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'deep': {'Lemma': 'deep', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'differ': {'Lemma': 'diff', 'POS': 'ADJ', 'POS_adv': 'JJR', 'Category': 'DESC NOUN',
		                                   'POS_adv_desc': 'adjective, comparative'},
		                        'direct': {'Lemma': 'direct', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                   'POS_adv_desc': 'adjective'},
		                        'dry': {'Lemma': 'dry', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                'POS_adv_desc': 'adjective'},
		                        'enough': {'Lemma': 'enough', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                   'POS_adv_desc': 'adjective'},
		                        'fast': {'Lemma': 'fast', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'few': {'Lemma': 'few', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                'POS_adv_desc': 'adjective'},
		                        'final': {'Lemma': 'final', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                  'POS_adv_desc': 'adjective'},
		                        'fine': {'Lemma': 'fine', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'free': {'Lemma': 'free', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'front': {'Lemma': 'front', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                  'POS_adv_desc': 'adjective'},
		                        'full': {'Lemma': 'full', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'good': {'Lemma': 'good', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'great': {'Lemma': 'great', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                  'POS_adv_desc': 'adjective'},
		                        'green': {'Lemma': 'green', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'COLOR',
		                                  'POS_adv_desc': 'adjective'},
		                        'hard': {'Lemma': 'hard', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'high': {'Lemma': 'high', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'hot': {'Lemma': 'hot', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                'POS_adv_desc': 'adjective'},
		                        'large': {'Lemma': 'large', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                  'POS_adv_desc': 'adjective'},
		                        'last': {'Lemma': 'last', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'late': {'Lemma': 'late', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'less': {'Lemma': 'less', 'POS': 'ADJ', 'POS_adv': 'JJR', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective, comparative'},
		                        'little': {'Lemma': 'little', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                   'POS_adv_desc': 'adjective'},
		                        'long': {'Lemma': 'long', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'low': {'Lemma': 'low', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                'POS_adv_desc': 'adjective'},
		                        'many': {'Lemma': 'many', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'mean': {'Lemma': 'mean', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'miss': {'Lemma': 'miss', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'more': {'Lemma': 'more', 'POS': 'ADJ', 'POS_adv': 'JJR', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective, comparative'},
		                        'much': {'Lemma': 'much', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'new': {'Lemma': 'new', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                'POS_adv_desc': 'adjective'},
		                        'old': {'Lemma': 'old', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                'POS_adv_desc': 'adjective'},
		                        'open': {'Lemma': 'open', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'other': {'Lemma': 'other', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                  'POS_adv_desc': 'adjective'},
		                        'own': {'Lemma': 'own', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                'POS_adv_desc': 'adjective'},
		                        'possible': {'Lemma': 'possible', 'POS': 'ADJ', 'POS_adv': 'JJ',
		                                     'Category': 'DESC NOUN', 'POS_adv_desc': 'adjective'},
		                        'quick': {'Lemma': 'quick', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                  'POS_adv_desc': 'adjective'},
		                        'ready': {'Lemma': 'ready', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                  'POS_adv_desc': 'adjective'},
		                        'real': {'Lemma': 'real', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'red': {'Lemma': 'red', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'COLOR',
		                                'POS_adv_desc': 'adjective'},
		                        'same': {'Lemma': 'same', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'second': {'Lemma': 'second', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                   'POS_adv_desc': 'adjective'},
		                        'several': {'Lemma': 'several', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                    'POS_adv_desc': 'adjective'},
		                        'short': {'Lemma': 'short', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                  'POS_adv_desc': 'adjective'},
		                        'simple': {'Lemma': 'simple', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                   'POS_adv_desc': 'adjective'},
		                        'slow': {'Lemma': 'slow', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'small': {'Lemma': 'small', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                  'POS_adv_desc': 'adjective'},
		                        'special': {'Lemma': 'special', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                    'POS_adv_desc': 'adjective'},
		                        'strong': {'Lemma': 'strong', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                   'POS_adv_desc': 'adjective'},
		                        'such': {'Lemma': 'such', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'ten': {'Lemma': 'ten', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                'POS_adv_desc': 'adjective'},
		                        'top': {'Lemma': 'top', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                'POS_adv_desc': 'adjective'},
		                        'true': {'Lemma': 'true', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'usual': {'Lemma': 'usual', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                  'POS_adv_desc': 'adjective'},
		                        'warm': {'Lemma': 'warm', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                 'POS_adv_desc': 'adjective'},
		                        'white': {'Lemma': 'white', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'COLOR',
		                                  'POS_adv_desc': 'adjective'},
		                        'whole': {'Lemma': 'whole', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                  'POS_adv_desc': 'adjective'},
		                        'wonder': {'Lemma': 'wonder', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                   'POS_adv_desc': 'adjective'},
		                        'young': {'Lemma': 'young', 'POS': 'ADJ', 'POS_adv': 'JJ', 'Category': 'DESC NOUN',
		                                  'POS_adv_desc': 'adjective'},
		                        'The': {'Lemma': 'the', 'POS': 'DET', 'POS_adv': 'DT', 'Category': 'DETERMINER',
		                                'POS_adv_desc': 'determiner'},
		                        'This': {'Lemma': 'this', 'POS': 'DET', 'POS_adv': 'DT', 'Category': 'DETERMINER',
		                                 'POS_adv_desc': 'determiner'},
		                        'all': {'Lemma': 'all', 'POS': 'DET', 'POS_adv': 'DT', 'Category': 'DETERMINER',
		                                'POS_adv_desc': 'determiner'},
		                        'both': {'Lemma': 'both', 'POS': 'DET', 'POS_adv': 'DT', 'Category': 'DETERMINER',
		                                 'POS_adv_desc': 'determiner'},
		                        'each': {'Lemma': 'each', 'POS': 'DET', 'POS_adv': 'DT', 'Category': 'DETERMINER',
		                                 'POS_adv_desc': 'determiner'},
		                        'every': {'Lemma': 'every', 'POS': 'DET', 'POS_adv': 'DT', 'Category': 'DETERMINER',
		                                  'POS_adv_desc': 'determiner'},
		                        'some': {'Lemma': 'some', 'POS': 'DET', 'POS_adv': 'DT', 'Category': 'DETERMINER',
		                                 'POS_adv_desc': 'determiner'},
		                        'that': {'Lemma': 'that', 'POS': 'DET', 'POS_adv': 'DT', 'Category': 'DETERMINER',
		                                 'POS_adv_desc': 'determiner'},
		                        'the': {'Lemma': 'the', 'POS': 'DET', 'POS_adv': 'DT', 'Category': 'DETERMINER',
		                                'POS_adv_desc': 'determiner'},
		                        'these': {'Lemma': 'these', 'POS': 'DET', 'POS_adv': 'DT', 'Category': 'DETERMINER',
		                                  'POS_adv_desc': 'determiner'},
		                        'this': {'Lemma': 'this', 'POS': 'DET', 'POS_adv': 'DT', 'Category': 'DETERMINER',
		                                 'POS_adv_desc': 'determiner'},
		                        'those': {'Lemma': 'those', 'POS': 'DET', 'POS_adv': 'DT', 'Category': 'DETERMINER',
		                                  'POS_adv_desc': 'determiner'},
		                        'which': {'Lemma': 'which', 'POS': 'DET', 'POS_adv': 'WDT', 'Category': 'DETERMINER',
		                                  'POS_adv_desc': 'wh-determiner'},
		                        'In': {'Lemma': 'in', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                               'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'about': {'Lemma': 'about', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                                  'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'after': {'Lemma': 'after', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                                  'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'against': {'Lemma': 'against', 'POS': 'ADP', 'POS_adv': 'IN',
		                                    'Category': 'CONJUNCTION',
		                                    'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'among': {'Lemma': 'among', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                                  'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'as': {'Lemma': 'as', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                               'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'at': {'Lemma': 'at', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                               'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'between': {'Lemma': 'between', 'POS': 'ADP', 'POS_adv': 'IN',
		                                    'Category': 'CONJUNCTION',
		                                    'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'by': {'Lemma': 'by', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                               'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'during': {'Lemma': 'during', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                                   'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'for': {'Lemma': 'for', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                                'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'from': {'Lemma': 'from', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                                 'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'in': {'Lemma': 'in', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                               'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'of': {'Lemma': 'of', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                               'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'on': {'Lemma': 'on', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                               'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'to': {'Lemma': 'to', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                               'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'under': {'Lemma': 'under', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                                  'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'until': {'Lemma': 'until', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                                  'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'with': {'Lemma': 'with', 'POS': 'ADP', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                                 'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'and': {'Lemma': 'and', 'POS': 'CCONJ', 'POS_adv': 'CC', 'Category': 'CONJUNCTION',
		                                'POS_adv_desc': 'conjunction, coordinating'},
		                        'but': {'Lemma': 'but', 'POS': 'CCONJ', 'POS_adv': 'CC', 'Category': 'CONJUNCTION',
		                                'POS_adv_desc': 'conjunction, coordinating'},
		                        'or': {'Lemma': 'or', 'POS': 'CCONJ', 'POS_adv': 'CC', 'Category': 'CONJUNCTION',
		                               'POS_adv_desc': 'conjunction, coordinating'},
		                        'A': {'Lemma': 'a', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                              'POS_adv_desc': 'interjection'},
		                        'a': {'Lemma': 'a', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                              'POS_adv_desc': 'interjection'},
		                        'any': {'Lemma': 'any', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                                'POS_adv_desc': 'interjection'},
		                        'ease': {'Lemma': 'ease', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                                 'POS_adv_desc': 'interjection'},
		                        'got': {'Lemma': 'got', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                                'POS_adv_desc': 'interjection'},
		                        'left': {'Lemma': 'left', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                                 'POS_adv_desc': 'interjection'},
		                        'like': {'Lemma': 'like', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                                 'POS_adv_desc': 'interjection'},
		                        'may': {'Lemma': 'may', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                                'POS_adv_desc': 'interjection'},
		                        'my': {'Lemma': 'my', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                               'POS_adv_desc': 'interjection'},
		                        'no': {'Lemma': 'no', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                               'POS_adv_desc': 'interjection'},
		                        'now': {'Lemma': 'now', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                                'POS_adv_desc': 'interjection'},
		                        'oh': {'Lemma': 'oh', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                               'POS_adv_desc': 'interjection'},
		                        'right': {'Lemma': 'right', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                                  'POS_adv_desc': 'interjection'},
		                        'sure': {'Lemma': 'sure', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                                 'POS_adv_desc': 'interjection'},
		                        'than': {'Lemma': 'than', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                                 'POS_adv_desc': 'interjection'},
		                        'though': {'Lemma': 'though', 'POS': 'INTJ', 'POS_adv': 'UH',
		                                   'Category': 'INTERJECTION', 'POS_adv_desc': 'interjection'},
		                        'up': {'Lemma': 'up', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                               'POS_adv_desc': 'interjection'},
		                        'well': {'Lemma': 'well', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                                 'POS_adv_desc': 'interjection'},
		                        'yes': {'Lemma': 'yes', 'POS': 'INTJ', 'POS_adv': 'UH', 'Category': 'INTERJECTION',
		                                'POS_adv_desc': 'interjection'},
		                        'Are': {'Lemma': 'be', 'POS': 'AUX', 'POS_adv': 'VBP', 'Category': '',
		                                'POS_adv_desc': 'verb, non-3rd person singular present'},
		                        'am': {'Lemma': 'am', 'POS': 'AUX', 'POS_adv': 'UH', 'Category': '',
		                               'POS_adv_desc': 'interjection'},
		                        'are': {'Lemma': 'be', 'POS': 'AUX', 'POS_adv': 'VBP', 'Category': '',
		                                'POS_adv_desc': 'verb, non-3rd person singular present'},
		                        'could': {'Lemma': 'could', 'POS': 'AUX', 'POS_adv': 'MD', 'Category': '',
		                                  'POS_adv_desc': 'verb, modal auxiliary'},
		                        'is': {'Lemma': 'be', 'POS': 'AUX', 'POS_adv': 'VBZ', 'Category': '',
		                               'POS_adv_desc': 'verb, 3rd person singular present'},
		                        'was': {'Lemma': 'be', 'POS': 'AUX', 'POS_adv': 'VBD', 'Category': '',
		                                'POS_adv_desc': 'verb, past tense'},
		                        'were': {'Lemma': 'be', 'POS': 'AUX', 'POS_adv': 'VBD', 'Category': '',
		                                 'POS_adv_desc': 'verb, past tense'},
		                        'would': {'Lemma': 'would', 'POS': 'AUX', 'POS_adv': 'MD', 'Category': '',
		                                  'POS_adv_desc': 'verb, modal auxiliary'},
		                        'I': {'Lemma': 'I', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                              'POS_adv_desc': 'pronoun, personal'},
		                        'he': {'Lemma': 'he', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                               'POS_adv_desc': 'pronoun, personal'},
		                        'her': {'Lemma': 'she', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                                'POS_adv_desc': 'pronoun, personal'},
		                        'him': {'Lemma': 'he', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                                'POS_adv_desc': 'pronoun, personal'},
		                        'his': {'Lemma': 'his', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                                'POS_adv_desc': 'pronoun, personal'},
		                        'it': {'Lemma': 'it', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                               'POS_adv_desc': 'pronoun, personal'},
		                        'me': {'Lemma': 'I', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                               'POS_adv_desc': 'pronoun, personal'},
		                        'nothing': {'Lemma': 'nothing', 'POS': 'PRON', 'POS_adv': 'NN', 'Category': 'PRONOUN',
		                                    'POS_adv_desc': 'noun, singular or mass'},
		                        'our': {'Lemma': 'our', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                                'POS_adv_desc': 'pronoun, personal'},
		                        'self': {'Lemma': 'self', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                                 'POS_adv_desc': 'pronoun, personal'},
		                        'she': {'Lemma': 'she', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                                'POS_adv_desc': 'pronoun, personal'},
		                        'their': {'Lemma': 'their', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                                  'POS_adv_desc': 'pronoun, personal'},
		                        'them': {'Lemma': 'they', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                                 'POS_adv_desc': 'pronoun, personal'},
		                        'they': {'Lemma': 'they', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                                 'POS_adv_desc': 'pronoun, personal'},
		                        'us': {'Lemma': 'we', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                               'POS_adv_desc': 'pronoun, personal'},
		                        'we': {'Lemma': 'we', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                               'POS_adv_desc': 'pronoun, personal'},
		                        'what': {'Lemma': 'what', 'POS': 'PRON', 'POS_adv': 'WP', 'Category': 'PRONOUN',
		                                 'POS_adv_desc': 'wh-pronoun, personal'},
		                        'who': {'Lemma': 'who', 'POS': 'PRON', 'POS_adv': 'WP', 'Category': 'PRONOUN',
		                                'POS_adv_desc': 'wh-pronoun, personal'},
		                        'you': {'Lemma': 'you', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                                'POS_adv_desc': 'pronoun, personal'},
		                        'your': {'Lemma': 'your', 'POS': 'PRON', 'POS_adv': 'PRP', 'Category': 'PRONOUN',
		                                 'POS_adv_desc': 'pronoun, personal'},
		                        'add': {'Lemma': 'add', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, base form'},
		                        'appear': {'Lemma': 'appear', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                   'POS_adv_desc': 'verb, base form'},
		                        'ask': {'Lemma': 'ask', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, base form'},
		                        'be': {'Lemma': 'be', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                               'POS_adv_desc': 'verb, base form'},
		                        'been': {'Lemma': 'be', 'POS': 'VERB', 'POS_adv': 'VBN', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, past participle'},
		                        'began': {'Lemma': 'begin', 'POS': 'VERB', 'POS_adv': 'VBN', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, past participle'},
		                        'begin': {'Lemma': 'begin', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'bring': {'Lemma': 'bring', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'brought': {'Lemma': 'bring', 'POS': 'VERB', 'POS_adv': 'VBN', 'Category': 'ACTION',
		                                    'POS_adv_desc': 'verb, past participle'},
		                        'build': {'Lemma': 'build', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'call': {'Lemma': 'call', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'came': {'Lemma': 'come', 'POS': 'VERB', 'POS_adv': 'VBD', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, past tense'},
		                        'carry': {'Lemma': 'carry', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'check': {'Lemma': 'check', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'come': {'Lemma': 'come', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'cry': {'Lemma': 'cry', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, base form'},
		                        'decide': {'Lemma': 'decide', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                   'POS_adv_desc': 'verb, base form'},
		                        'develop': {'Lemma': 'develop', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                    'POS_adv_desc': 'verb, base form'},
		                        'did': {'Lemma': 'do', 'POS': 'VERB', 'POS_adv': 'VBD', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, past tense'},
		                        'do': {'Lemma': 'do', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                               'POS_adv_desc': 'verb, base form'},
		                        'does': {'Lemma': 'do', 'POS': 'VERB', 'POS_adv': 'VBZ', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, 3rd person singular present'},
		                        'done': {'Lemma': 'do', 'POS': 'VERB', 'POS_adv': 'VBN', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, past participle'},
		                        'draw': {'Lemma': 'draw', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'drive': {'Lemma': 'drive', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'eat': {'Lemma': 'eat', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, base form'},
		                        'feel': {'Lemma': 'feel', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'find': {'Lemma': 'find', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'fly': {'Lemma': 'fly', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, base form'},
		                        'follow': {'Lemma': 'follow', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                   'POS_adv_desc': 'verb, base form'},
		                        'found': {'Lemma': 'find', 'POS': 'VERB', 'POS_adv': 'VBN', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, past participle'},
		                        'gave': {'Lemma': 'give', 'POS': 'VERB', 'POS_adv': 'VBN', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, past participle'},
		                        'get': {'Lemma': 'get', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, base form'},
		                        'give': {'Lemma': 'give', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'go': {'Lemma': 'go', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                               'POS_adv_desc': 'verb, base form'},
		                        'govern': {'Lemma': 'govern', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                   'POS_adv_desc': 'verb, base form'},
		                        'grow': {'Lemma': 'grow', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'had': {'Lemma': 'have', 'POS': 'VERB', 'POS_adv': 'VBN', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, past participle'},
		                        'happen': {'Lemma': 'happen', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                   'POS_adv_desc': 'verb, base form'},
		                        'has': {'Lemma': 'have', 'POS': 'VERB', 'POS_adv': 'VBZ', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, 3rd person singular present'},
		                        'have': {'Lemma': 'have', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'hear': {'Lemma': 'hear', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'heard': {'Lemma': 'hear', 'POS': 'VERB', 'POS_adv': 'VBN', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, past participle'},
		                        'help': {'Lemma': 'help', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'hold': {'Lemma': 'hold', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'keep': {'Lemma': 'keep', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'knew': {'Lemma': 'know', 'POS': 'VERB', 'POS_adv': 'VBN', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, past participle'},
		                        'know': {'Lemma': 'know', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'laugh': {'Lemma': 'laugh', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'lead': {'Lemma': 'lead', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'learn': {'Lemma': 'learn', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'leave': {'Lemma': 'leave', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'let': {'Lemma': 'let', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, base form'},
		                        'listen': {'Lemma': 'listen', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                   'POS_adv_desc': 'verb, base form'},
		                        'live': {'Lemma': 'live', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'look': {'Lemma': 'look', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'made': {'Lemma': 'make', 'POS': 'VERB', 'POS_adv': 'VBN', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, past participle'},
		                        'make': {'Lemma': 'make', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'move': {'Lemma': 'move', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'must': {'Lemma': 'must', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'notice': {'Lemma': 'notice', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                   'POS_adv_desc': 'verb, base form'},
		                        'play': {'Lemma': 'play', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'pull': {'Lemma': 'pull', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'put': {'Lemma': 'put', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, base form'},
		                        'reach': {'Lemma': 'reach', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'read': {'Lemma': 'read', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'remember': {'Lemma': 'remember', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                     'POS_adv_desc': 'verb, base form'},
		                        'run': {'Lemma': 'run', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, base form'},
		                        'said': {'Lemma': 'say', 'POS': 'VERB', 'POS_adv': 'VBD', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, past tense'},
		                        'saw': {'Lemma': 'see', 'POS': 'VERB', 'POS_adv': 'VBN', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, past participle'},
		                        'say': {'Lemma': 'say', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, base form'},
		                        'see': {'Lemma': 'see', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, base form'},
		                        'seem': {'Lemma': 'seem', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'serve': {'Lemma': 'serve', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'should': {'Lemma': 'should', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                   'POS_adv_desc': 'verb, base form'},
		                        'sit': {'Lemma': 'sit', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, base form'},
		                        'snowing': {'Lemma': 'snow', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                    'POS_adv_desc': 'verb, base form'},
		                        'stand': {'Lemma': 'stand', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'start': {'Lemma': 'start', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'stay': {'Lemma': 'stay', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'stood': {'Lemma': 'stand', 'POS': 'VERB', 'POS_adv': 'VBN', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, past participle'},
		                        'stop': {'Lemma': 'stop', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'take': {'Lemma': 'took', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'talk': {'Lemma': 'talk', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'teach': {'Lemma': 'teach', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'tell': {'Lemma': 'tell', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'think': {'Lemma': 'think', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'thought': {'Lemma': 'think', 'POS': 'VERB', 'POS_adv': 'VBN', 'Category': 'ACTION',
		                                    'POS_adv_desc': 'verb, past participle'},
		                        'told': {'Lemma': 'tell', 'POS': 'VERB', 'POS_adv': 'VBN', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, past participle'},
		                        'try': {'Lemma': 'try', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                'POS_adv_desc': 'verb, base form'},
		                        'wait': {'Lemma': 'wait', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'want': {'Lemma': 'want', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, base form'},
		                        'watch': {'Lemma': 'watch', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'went': {'Lemma': 'go', 'POS': 'VERB', 'POS_adv': 'VBD', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, past tense'},
		                        'write': {'Lemma': 'write', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, base form'},
		                        'written': {'Lemma': 'write', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, past tense form'},
		                        'wrote': {'Lemma': 'write', 'POS': 'VERB', 'POS_adv': 'VB', 'Category': 'ACTION',
		                                  'POS_adv_desc': 'verb, past tense'},
		                        'Thousand': {'Lemma': 'thousand', 'POS': 'NUM', 'POS_adv': 'CD', 'Category': 'NUMBER',
		                                     'POS_adv_desc': 'cardinal number'},
		                        'eight': {'Lemma': 'eight', 'POS': 'NUM', 'POS_adv': 'CD', 'Category': 'NUMBER',
		                                  'POS_adv_desc': 'cardinal number'},
		                        'five': {'Lemma': 'five', 'POS': 'NUM', 'POS_adv': 'CD', 'Category': 'NUMBER',
		                                 'POS_adv_desc': 'cardinal number'},
		                        'four': {'Lemma': 'four', 'POS': 'NUM', 'POS_adv': 'CD', 'Category': 'NUMBER',
		                                 'POS_adv_desc': 'cardinal number'},
		                        'hundred': {'Lemma': 'hundred', 'POS': 'NUM', 'POS_adv': 'CD', 'Category': 'NUMBER',
		                                    'POS_adv_desc': 'cardinal number'},
		                        'nine': {'Lemma': 'nine', 'POS': 'NUM', 'POS_adv': 'CD', 'Category': 'NUMBER',
		                                 'POS_adv_desc': 'cardinal number'},
		                        'one': {'Lemma': 'one', 'POS': 'NUM', 'POS_adv': 'CD', 'Category': 'NUMBER',
		                                'POS_adv_desc': 'cardinal number'},
		                        'seven': {'Lemma': 'seven', 'POS': 'NUM', 'POS_adv': 'CD', 'Category': 'NUMBER',
		                                  'POS_adv_desc': 'cardinal number'},
		                        'six': {'Lemma': 'six', 'POS': 'NUM', 'POS_adv': 'CD', 'Category': 'NUMBER',
		                                'POS_adv_desc': 'cardinal number'},
		                        'thousand': {'Lemma': 'thousand', 'POS': 'NUM', 'POS_adv': 'CD', 'Category': 'NUMBER',
		                                     'POS_adv_desc': 'cardinal number'},
		                        'three': {'Lemma': 'three', 'POS': 'NUM', 'POS_adv': 'CD', 'Category': 'NUMBER',
		                                  'POS_adv_desc': 'cardinal number'},
		                        'two': {'Lemma': 'two', 'POS': 'NUM', 'POS_adv': 'CD', 'Category': 'NUMBER',
		                                'POS_adv_desc': 'cardinal number'},
		                        'There': {'Lemma': 'there', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                  'POS_adv_desc': 'adverb'},
		                        'Where': {'Lemma': 'where', 'POS': 'ADV', 'POS_adv': 'WRB', 'Category': 'DESC VERB',
		                                  'POS_adv_desc': 'wh-adverb'},
		                        'above': {'Lemma': 'above', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                  'POS_adv_desc': 'adverb'},
		                        'again': {'Lemma': 'again', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                  'POS_adv_desc': 'adverb'},
		                        'also': {'Lemma': 'also', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb'},
		                        'always': {'Lemma': 'always', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                   'POS_adv_desc': 'adverb'},
		                        'back': {'Lemma': 'back', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb'},
		                        'before': {'Lemma': 'before', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                   'POS_adv_desc': 'adverb'},
		                        'behind': {'Lemma': 'behind', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                   'POS_adv_desc': 'adverb'},
		                        'best': {'Lemma': 'well', 'POS': 'ADV', 'POS_adv': 'RBS', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb, superlative'},
		                        'better': {'Lemma': 'well', 'POS': 'ADV', 'POS_adv': 'RBR', 'Category': 'DESC VERB',
		                                   'POS_adv_desc': 'adverb, comparative'},
		                        'close': {'Lemma': 'close', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                  'POS_adv_desc': 'adverb'},
		                        'down': {'Lemma': 'down', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb'},
		                        'early': {'Lemma': 'early', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                  'POS_adv_desc': 'adverb'},
		                        'east': {'Lemma': 'east', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DIRECTION',
		                                 'POS_adv_desc': 'adverb'},
		                        'even': {'Lemma': 'even', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb'},
		                        'ever': {'Lemma': 'ever', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb'},
		                        'far': {'Lemma': 'far', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                'POS_adv_desc': 'adverb'},
		                        'first': {'Lemma': 'first', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                  'POS_adv_desc': 'adverb'},
		                        'here': {'Lemma': 'here', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb'},
		                        'how': {'Lemma': 'how', 'POS': 'ADV', 'POS_adv': 'WRB', 'Category': 'DESC VERB',
		                                'POS_adv_desc': 'wh-adverb'},
		                        'just': {'Lemma': 'just', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb'},
		                        'kind': {'Lemma': 'kind', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb'},
		                        'might': {'Lemma': 'might', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                  'POS_adv_desc': 'adverb'},
		                        'most': {'Lemma': 'most', 'POS': 'ADV', 'POS_adv': 'RBS', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb, superlative'},
		                        'near': {'Lemma': 'near', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb'},
		                        'never': {'Lemma': 'never', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                  'POS_adv_desc': 'adverb'},
		                        'next': {'Lemma': 'next', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb'},
		                        'north': {'Lemma': 'north', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DIRECTION',
		                                  'POS_adv_desc': 'adverb'},
		                        'off': {'Lemma': 'off', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                'POS_adv_desc': 'adverb'},
		                        'often': {'Lemma': 'often', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                  'POS_adv_desc': 'adverb'},
		                        'once': {'Lemma': 'once', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb'},
		                        'only': {'Lemma': 'only', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb'},
		                        'out': {'Lemma': 'out', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                'POS_adv_desc': 'adverb'},
		                        'over': {'Lemma': 'over', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb'},
		                        'perhaps': {'Lemma': 'perhaps', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                    'POS_adv_desc': 'adverb'},
		                        'round': {'Lemma': 'round', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                  'POS_adv_desc': 'adverb'},
		                        'so': {'Lemma': 'so', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                               'POS_adv_desc': 'adverb'},
		                        'soon': {'Lemma': 'soon', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb'},
		                        'still': {'Lemma': 'still', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                  'POS_adv_desc': 'adverb'},
		                        'then': {'Lemma': 'then', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb'},
		                        'there': {'Lemma': 'there', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                  'POS_adv_desc': 'adverb'},
		                        'through': {'Lemma': 'through', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                    'POS_adv_desc': 'adverb'},
		                        'together': {'Lemma': 'together', 'POS': 'ADV', 'POS_adv': 'RB',
		                                     'Category': 'DESC VERB', 'POS_adv_desc': 'adverb'},
		                        'too': {'Lemma': 'too', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                'POS_adv_desc': 'adverb'},
		                        'toward': {'Lemma': 'toward', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                   'POS_adv_desc': 'adverb'},
		                        'very': {'Lemma': 'very', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'adverb'},
		                        'west': {'Lemma': 'west', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DIRECTION',
		                                 'POS_adv_desc': 'adverb'},
		                        'when': {'Lemma': 'when', 'POS': 'ADV', 'POS_adv': 'WRB', 'Category': 'DESC VERB',
		                                 'POS_adv_desc': 'wh-adverb'},
		                        'where': {'Lemma': 'where', 'POS': 'ADV', 'POS_adv': 'WRB', 'Category': 'DESC VERB',
		                                  'POS_adv_desc': 'wh-adverb'},
		                        'while': {'Lemma': 'while', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                  'POS_adv_desc': 'adverb'},
		                        'why': {'Lemma': 'why', 'POS': 'ADV', 'POS_adv': 'WRB', 'Category': 'DESC VERB',
		                                'POS_adv_desc': 'wh-adverb'},
		                        'yet': {'Lemma': 'yet', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DESC VERB',
		                                'POS_adv_desc': 'adverb'},
		                        'Children': {'Lemma': 'child', 'POS': 'NOUN', 'POS_adv': 'NNS', 'Category': 'PERSON',
		                                     'POS_adv_desc': 'noun, plural'},
		                        'Town': {'Lemma': 'town', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'PLACE',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'act': {'Lemma': 'act', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'age': {'Lemma': 'age', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'DESC',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'air': {'Lemma': 'air', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'animal': {'Lemma': 'animal', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'THING',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'answer': {'Lemma': 'answer', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'area': {'Lemma': 'area', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'base': {'Lemma': 'base', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'beauty': {'Lemma': 'beauty', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'DESC',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'bed': {'Lemma': 'bed', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'bird': {'Lemma': 'bird', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'THING',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'boat': {'Lemma': 'boat', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'body': {'Lemma': 'body', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'book': {'Lemma': 'book', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'box': {'Lemma': 'box', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'boy': {'Lemma': 'boy', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'PERSON',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'car': {'Lemma': 'car', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'care': {'Lemma': 'care', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'cause': {'Lemma': 'cause', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'center': {'Lemma': 'center', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'change': {'Lemma': 'change', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'children': {'Lemma': 'child', 'POS': 'NOUN', 'POS_adv': 'NNS', 'Category': 'PERSON',
		                                     'POS_adv_desc': 'noun, plural'},
		                        'city': {'Lemma': 'city', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'PLACE',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'class': {'Lemma': 'class', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'color': {'Lemma': 'color', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'contain': {'Lemma': 'contain', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                    'POS_adv_desc': 'noun, singular or mass'},
		                        'country': {'Lemma': 'country', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                    'POS_adv_desc': 'noun, singular or mass'},
		                        'course': {'Lemma': 'course', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'cover': {'Lemma': 'cover', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'cross': {'Lemma': 'cross', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'cut': {'Lemma': 'cut', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'day': {'Lemma': 'day', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'TIME',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'dog': {'Lemma': 'dog', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'THING',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'door': {'Lemma': 'door', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'earth': {'Lemma': 'earth', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'end': {'Lemma': 'end', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'example': {'Lemma': 'example', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                    'POS_adv_desc': 'noun, singular or mass'},
		                        'eye': {'Lemma': 'eye', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'face': {'Lemma': 'face', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'fact': {'Lemma': 'fact', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'fall': {'Lemma': 'fall', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'Fall': {'Lemma': 'Fall', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'TIME',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'family': {'Lemma': 'family', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'PERSON',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'farm': {'Lemma': 'farm', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'father': {'Lemma': 'father', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'PERSON',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'feet': {'Lemma': 'foot', 'POS': 'NOUN', 'POS_adv': 'NNS', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, plural'},
		                        'field': {'Lemma': 'field', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'figure': {'Lemma': 'figure', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'PERSON',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'fill': {'Lemma': 'fill', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'fire': {'Lemma': 'fire', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'fish': {'Lemma': 'fish', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'THING',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'food': {'Lemma': 'food', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'foot': {'Lemma': 'foot', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'force': {'Lemma': 'force', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'form': {'Lemma': 'form', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'friend': {'Lemma': 'friend', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'PERSON',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'game': {'Lemma': 'game', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'girl': {'Lemma': 'girl', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'PERSON',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'gold': {'Lemma': 'gold', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'ground': {'Lemma': 'ground', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'group': {'Lemma': 'group', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'half': {'Lemma': 'half', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'hand': {'Lemma': 'hand', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'head': {'Lemma': 'head', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'heat': {'Lemma': 'heat', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'home': {'Lemma': 'home', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'horse': {'Lemma': 'horse', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'THING',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'hour': {'Lemma': 'hour', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'TIME',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'house': {'Lemma': 'house', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'idea': {'Lemma': 'idea', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'inch': {'Lemma': 'inch', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'interest': {'Lemma': 'interest', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                     'POS_adv_desc': 'noun, singular or mass'},
		                        'island': {'Lemma': 'island', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'king': {'Lemma': 'king', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'PERSON',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'land': {'Lemma': 'land', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'language': {'Lemma': 'language', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                     'POS_adv_desc': 'noun, singular or mass'},
		                        'lay': {'Lemma': 'lay', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'letter': {'Lemma': 'letter', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'life': {'Lemma': 'life', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'light': {'Lemma': 'light', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'line': {'Lemma': 'line', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'list': {'Lemma': 'list', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'lot': {'Lemma': 'lot', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'love': {'Lemma': 'love', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'machine': {'Lemma': 'machine', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                    'POS_adv_desc': 'noun, singular or mass'},
		                        'man': {'Lemma': 'man', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'PERSON',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'map': {'Lemma': 'map', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'mark': {'Lemma': 'mark', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'measure': {'Lemma': 'measure', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                    'POS_adv_desc': 'noun, singular or mass'},
		                        'men': {'Lemma': 'man', 'POS': 'NOUN', 'POS_adv': 'NNS', 'Category': 'PERSON',
		                                'POS_adv_desc': 'noun, plural'},
		                        'mile': {'Lemma': 'mile', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'mind': {'Lemma': 'mind', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'minute': {'Lemma': 'minute', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'TIME',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'money': {'Lemma': 'money', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'moon': {'Lemma': 'moon', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'morning': {'Lemma': 'morning', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'TIME',
		                                    'POS_adv_desc': 'noun, singular or mass'},
		                        'mother': {'Lemma': 'mother', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'PERSON',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'mountain': {'Lemma': 'mountain', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                     'POS_adv_desc': 'noun, singular or mass'},
		                        'music': {'Lemma': 'music', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'name': {'Lemma': 'name', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'need': {'Lemma': 'need', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'night': {'Lemma': 'night', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'TIME',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'note': {'Lemma': 'note', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'noun': {'Lemma': 'noun', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'number': {'Lemma': 'number', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'numeral': {'Lemma': 'numeral', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                    'POS_adv_desc': 'noun, singular or mass'},
		                        'object': {'Lemma': 'object', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'order': {'Lemma': 'order', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'page': {'Lemma': 'page', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'paper': {'Lemma': 'paper', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'part': {'Lemma': 'part', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'pass': {'Lemma': 'pass', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'pattern': {'Lemma': 'pattern', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                    'POS_adv_desc': 'noun, singular or mass'},
		                        'people': {'Lemma': 'people', 'POS': 'NOUN', 'POS_adv': 'NNS', 'Category': 'PERSON',
		                                   'POS_adv_desc': 'noun, plural'},
		                        'person': {'Lemma': 'person', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'PERSON',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'picture': {'Lemma': 'picture', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                    'POS_adv_desc': 'noun, singular or mass'},
		                        'piece': {'Lemma': 'piece', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'place': {'Lemma': 'place', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'plain': {'Lemma': 'plain', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'plan': {'Lemma': 'plan', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'plane': {'Lemma': 'plane', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'plant': {'Lemma': 'plant', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'point': {'Lemma': 'point', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'port': {'Lemma': 'port', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'pose': {'Lemma': 'pose', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'pound': {'Lemma': 'pound', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'power': {'Lemma': 'power', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'press': {'Lemma': 'press', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'problem': {'Lemma': 'problem', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                    'POS_adv_desc': 'noun, singular or mass'},
		                        'produce': {'Lemma': 'produce', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                    'POS_adv_desc': 'noun, singular or mass'},
		                        'product': {'Lemma': 'product', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                    'POS_adv_desc': 'noun, singular or mass'},
		                        'question': {'Lemma': 'question', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                     'POS_adv_desc': 'noun, singular or mass'},
		                        'rain': {'Lemma': 'rain', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'record': {'Lemma': 'record', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'rest': {'Lemma': 'rest', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'river': {'Lemma': 'river', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'road': {'Lemma': 'road', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'rock': {'Lemma': 'rock', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'room': {'Lemma': 'room', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'rule': {'Lemma': 'rule', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'school': {'Lemma': 'school', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'science': {'Lemma': 'science', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                    'POS_adv_desc': 'noun, singular or mass'},
		                        'sea': {'Lemma': 'sea', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'sentence': {'Lemma': 'sentence', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                     'POS_adv_desc': 'noun, singular or mass'},
		                        'set': {'Lemma': 'set', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'shape': {'Lemma': 'shape', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'ship': {'Lemma': 'ship', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'show': {'Lemma': 'show', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'side': {'Lemma': 'side', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'sing': {'Lemma': 'sing', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'size': {'Lemma': 'size', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'sleep': {'Lemma': 'sleep', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'song': {'Lemma': 'song', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'sound': {'Lemma': 'sound', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'south': {'Lemma': 'south', 'POS': 'ADV', 'POS_adv': 'RB', 'Category': 'DIRECTION',
		                                  'POS_adv_desc': 'adverb'},
		                        'spell': {'Lemma': 'spell', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'star': {'Lemma': 'star', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'state': {'Lemma': 'state', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'step': {'Lemma': 'step', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'story': {'Lemma': 'story', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'street': {'Lemma': 'street', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'study': {'Lemma': 'study', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'sun': {'Lemma': 'sun', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'surface': {'Lemma': 'surface', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                    'POS_adv_desc': 'noun, singular or mass'},
		                        'table': {'Lemma': 'table', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'tail': {'Lemma': 'tail', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'test': {'Lemma': 'test', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'thing': {'Lemma': 'thing', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'time': {'Lemma': 'time', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'town': {'Lemma': 'town', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'PLACE',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'travel': {'Lemma': 'travel', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'tree': {'Lemma': 'tree', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'turn': {'Lemma': 'turn', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'unit': {'Lemma': 'unit', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'use': {'Lemma': 'use', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'voice': {'Lemma': 'voice', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'walk': {'Lemma': 'walk', 'POS': 'VERB', 'POS_adv': 'NN', 'Category': 'ACTION',
		                                 'POS_adv_desc': 'verb, verb things'},
		                        'war': {'Lemma': 'war', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'water': {'Lemma': 'water', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'MULTI',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'way': {'Lemma': 'way', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                'POS_adv_desc': 'noun, singular or mass'},
		                        'week': {'Lemma': 'week', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'weight': {'Lemma': 'weight', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                   'POS_adv_desc': 'noun, singular or mass'},
		                        'wheel': {'Lemma': 'wheel', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'wind': {'Lemma': 'wind', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'wood': {'Lemma': 'wood', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'word': {'Lemma': 'word', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'work': {'Lemma': 'work', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'world': {'Lemma': 'world', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                  'POS_adv_desc': 'noun, singular or mass'},
		                        'year': {'Lemma': 'year', 'POS': 'NOUN', 'POS_adv': 'NN', 'Category': 'OBJECT',
		                                 'POS_adv_desc': 'noun, singular or mass'},
		                        'if': {'Lemma': 'if', 'POS': 'SCONJ', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                               'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'since': {'Lemma': 'since', 'POS': 'SCONJ', 'POS_adv': 'IN', 'Category': 'CONJUNCTION',
		                                  'POS_adv_desc': 'conjunction, subordinating or preposition'},
		                        'don’t': {'Lemma': 'n’t', 'POS': 'PART', 'POS_adv': 'RB', 'Category': '',
		                                  'POS_adv_desc': 'adverb'}}

