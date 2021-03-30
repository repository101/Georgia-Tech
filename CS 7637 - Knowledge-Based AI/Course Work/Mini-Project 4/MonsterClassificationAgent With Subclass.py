import numpy as np
import copy

# from plot_util import plot_single, plot_combined


class MonsterClassificationAgent:
	def solve(self, samples, new_monster):
		monster_subclass = SubMonsterClass()
		val = monster_subclass.solve(samples=samples, new_monster=new_monster)
		return val
			

class SubMonsterClass:
	def __init__(self):
		# If you want to do any initial processing, add it here.
		self.positive_examples = []
		self.positive_examples_encoded = None
		self.positive_traits = {}
		self.positive_traits_set = set()
		self.negative_examples = []
		self.negative_examples_encoded = None
		self.negative_traits = {}
		self.negative_traits_set = set()
		self.original_samples = None
		self.avg_positive = np.zeros(shape=(12,))
		self.avg_negative = np.zeros(shape=(12,))
		self.count = np.zeros(shape=(12,))
		self.count_dict = {}
		self.traits = [{} for _ in range(12)]
		self.trait_locations = {}
		self.num_traits = 12
		self.samples_encoded = None
		self.largest_value = 0
		self.trait_labels = []
		
	def calc_squared_error(self, data):
	
	
	def encode(self, sample):
		encoding = np.zeros(shape=(12,))
		for key, val in sample.items():
			# Get location of the trait
			trait_location = self.trait_locations[key]
			if bool(self.traits[trait_location]):
				encoding[trait_location] = self.traits[trait_location][val]
			else:
				encoding[trait_location] = val
		return encoding
	
	def setup(self, samples):
		self.positive_examples = []
		self.positive_examples_encoded = None
		self.positive_traits = {}
		self.positive_traits_set = set()
		self.negative_examples = []
		self.negative_examples_encoded = None
		self.negative_traits = {}
		self.negative_traits_set = set()
		self.original_samples = None
		self.avg_positive = np.zeros(shape=(12,))
		self.avg_negative = np.zeros(shape=(12,))
		self.count = np.zeros(shape=(12,))
		self.count_dict = {}
		self.traits = [{} for _ in range(12)]
		self.trait_locations = {}
		self.num_traits = 12
		self.samples_encoded = None
		self.largest_value = 0
		self.trait_labels = []
		
		self.original_samples = copy.deepcopy(samples)
		for i in range(len(samples)):
			if samples[i][1]:
				self.positive_examples.append({"Original": samples[i]})
				trait_idx = 0
				for key, val in samples[i][0].items():
					if val not in self.positive_traits_set:
						self.positive_traits_set.add(val)
					if key not in self.trait_labels:
						self.trait_labels.append(key)
					if isinstance(val, bool) or not isinstance(val, int):
						trait_count = self.count[trait_idx]
						self.trait_locations[key] = trait_idx
						if val not in self.traits[trait_idx]:
							self.count[trait_idx] += 1
							self.traits[trait_idx][val] = trait_count
						self.count_dict[key] = trait_count
						if trait_count > self.largest_value:
							self.largest_value = trait_count
					else:
						if val > self.largest_value:
							self.largest_value = val
					self.trait_locations[key] = trait_idx
					trait_idx += 1
					if key not in self.positive_traits:
						self.positive_traits[key] = [val]
					else:
						self.positive_traits[key].append(val)
			
			elif not samples[i][1]:
				self.negative_examples.append({"Original": samples[i]})
				trait_idx = 0
				for key, val in samples[i][0].items():
					if val not in self.negative_traits_set:
						self.negative_traits_set.add(val)
					if key not in self.trait_labels:
						self.trait_labels.append(key)
					if isinstance(val, bool) or not isinstance(val, int):
						trait_count = self.count[trait_idx]
						if val not in self.traits[trait_idx]:
							self.count[trait_idx] += 1
							self.traits[trait_idx][val] = trait_count
						self.count_dict[key] = trait_count
						if trait_count > self.largest_value:
							self.largest_value = trait_count
					else:
						if val > self.largest_value:
							self.largest_value = val
					self.trait_locations[key] = trait_idx
					trait_idx += 1
					if key not in self.negative_traits:
						self.negative_traits[key] = [val]
					else:
						self.negative_traits[key].append(val)
		for i in range(len(self.positive_examples)):
			self.positive_examples[i]["Encoded"] = self.encode(self.positive_examples[i]["Original"][0])
			if self.samples_encoded is None:
				self.samples_encoded = self.positive_examples[i]["Encoded"]
			else:
				self.samples_encoded = np.vstack((self.samples_encoded,
				                                  self.positive_examples[i]["Encoded"]))
			if self.positive_examples_encoded is None:
				self.positive_examples_encoded = self.positive_examples[i]["Encoded"]
			else:
				self.positive_examples_encoded = np.vstack((self.positive_examples_encoded,
				                                            self.positive_examples[i]["Encoded"]))
		
		for i in range(len(self.negative_examples)):
			self.negative_examples[i]["Encoded"] = self.encode(self.negative_examples[i]["Original"][0])
			if self.samples_encoded is None:
				self.samples_encoded = self.negative_examples[i]["Encoded"]
			else:
				self.samples_encoded = np.vstack((self.samples_encoded,
				                                  self.negative_examples[i]["Encoded"]))
			if self.negative_examples_encoded is None:
				self.negative_examples_encoded = self.negative_examples[i]["Encoded"]
			else:
				self.negative_examples_encoded = np.vstack((self.negative_examples_encoded,
				                                            self.negative_examples[i]["Encoded"]))
		
		# plot_combined(positive_examples_encoded=self.positive_examples_encoded,
		#               negative_examples_encoded=self.negative_examples_encoded,
		#               title="Combined Examples Compared",
		#               xlabel="Traits",
		#               ylabel="Encoded Value",
		#               trait_labels=self.trait_labels,
		#               y_limit=self.largest_value,
		#               save_name="Combined Examples Scatter Plot")
		#
		# plot_single(encoded_data=self.positive_examples_encoded,
		#             title="Positive Examples",
		#             xlabel="Traits",
		#             ylabel="Encoded Value",
		#             trait_labels=self.trait_labels,
		#             y_limit=self.largest_value,
		#             color="tab:green",
		#             marker="1",
		#             label="Positive Instances", type="single", save_file_name="Positive Examples Scatter Plot")
		# plot_single(encoded_data=self.negative_examples_encoded,
		#             title="Negative Examples",
		#             xlabel="Traits",
		#             ylabel="Encoded Value",
		#             trait_labels=self.trait_labels,
		#             y_limit=self.largest_value,
		#             color="tab:red",
		#             marker="2",
		#             label="Negative Instances", type="single", save_file_name="Negative Examples Scatter Plot")
		return
	
	def solve(self, samples, new_monster):
		self.setup(samples)
		likelihood_value_found = []
		temp_result = []
		for key, val in new_monster.items():
			pct_of_positive_instances = 0
			pct_of_negative_instances = 0
			if key in self.positive_traits:
				number_of_instances_consistent_with_value = sum(1 for i in self.positive_traits[key] if i == val)
				pct_of_positive_instances = number_of_instances_consistent_with_value / (
							len(self.negative_traits[key]) + len(self.positive_traits[key]))
			if key in self.negative_traits:
				number_of_instances_consistent_with_value = sum(1 for i in self.negative_traits[key] if i == val)
				pct_of_negative_instances = number_of_instances_consistent_with_value / (
							len(self.negative_traits[key]) + len(self.positive_traits[key]))
			if pct_of_positive_instances > pct_of_negative_instances and (
					pct_of_positive_instances - pct_of_negative_instances) > 0.1:
				temp_result.append("Positive")
			elif pct_of_negative_instances > pct_of_positive_instances and (
					pct_of_negative_instances - pct_of_positive_instances) > 0.1:
				temp_result.append("Negative")
			else:
				temp_result.append("EITHER")
			likelihood_value_found.append(
				{"Positive": pct_of_positive_instances, "Negative": pct_of_negative_instances})
		values = []
		for i in range(len(likelihood_value_found)):
			values.append(likelihood_value_found[i]["Positive"] * 1 + likelihood_value_found[i]["Negative"] * -1)
			self.avg_positive[i] = sum(self.positive_examples_encoded[:, i]) / len(self.positive_examples_encoded[:, i])
			self.avg_negative[i] = sum(self.negative_examples_encoded[:, i]) / len(self.negative_examples_encoded[:, i])
		for key, val in new_monster.items():
			if key in self.positive_traits:
				if val not in self.positive_traits[key]:
					return False
		return True
