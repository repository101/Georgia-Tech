import numpy as np
import copy

# from plot_util import plot_single, plot_combined


class MonsterClassificationAgent:
	def solve(self, samples, new_monster):
		#Add your code here!
		#
		#The first parameter to this method will be a labeled list of samples in the form of
		#a list of 2-tuples. The first item in each 2-tuple will be a dictionary representing
		#the parameters of a particular monster. The second item in each 2-tuple will be a
		#boolean indicating whether this is an example of this species or not.
		#
		#The second parameter will be a dictionary representing a newly observed monster.
		#
		#Your function should return True or False as a guess as to whether or not this new
		#monster is an instance of the same species as that represented by the list.
		monster_sub_class = MonsterSubClass()
		result = monster_sub_class.solve(samples=samples, new_monster=new_monster)
		for key, val in new_monster.items():
			if val not in monster_sub_class.positive_traits[key]:
				return False
		return True


class MonsterSubClass:
	def __init__(self):
		# If you want to do any initial processing, add it here.
		self.positive_examples = []
		self.positive_examples_encoded = None
		self.positive_traits = {}
		self.num_traits = 12
		self.positive_traits_set = set()
		self.negative_examples = []
		self.negative_examples_encoded = None
		self.negative_traits = {}
		self.negative_traits_set = set()
		self.original_samples = None
		self.avg_positive = np.zeros(shape=(self.num_traits,))
		self.avg_negative = np.zeros(shape=(self.num_traits,))
		self.count = np.zeros(shape=(self.num_traits,))
		self.count_dict = {}
		self.traits = [{} for _ in range(self.num_traits)]
		self.trait_counts = [{} for _ in range(self.num_traits)]
		self.trait_locations = {}
		self.samples_encoded = None
		self.samples_encoded_scaled = None
		self.samples_answers = None
		self.samples_answers_encoded = None
		self.largest_value = 0
		self.trait_labels = []

	@staticmethod
	def calc_mean(data):
		# https://dataaspirant.com/simple-linear-regression-python-without-any-machine-learning-libraries/
		return np.mean(data)

	def calc_variance(self, data):
		# https://dataaspirant.com/simple-linear-regression-python-without-any-machine-learning-libraries/
		data_mean = self.calc_mean(data=data)
		sse = [(data[i] - data_mean) ** 2 for i in range(len(data))]
		return np.sum(sse) / (len(data) - 1)

	def calc_covariance(self, data_x, data_y):
		# https://dataaspirant.com/simple-linear-regression-python-without-any-machine-learning-libraries/
		data_x_mean = self.calc_mean(data_x)
		data_y_mean = self.calc_mean(data_y)
		covariance = 0.0
		for i in range(len(data_x)):
			covariance += (data_x[i] - data_x_mean) * (data_y[i] - data_y_mean)
		return covariance / (len(data_x) - 1)

	@staticmethod
	def calc_rmse(data, predicted):
		square_error = 0
		for i in range(len(data)):
			square_error += (predicted[i] - data[i]) ** 2
		return square_error / len(data)

	def calc_linear_regression_coefficients(self, data_x, data_y):
		w1 = self.calc_covariance(data_x=data_x, data_y=data_y) / self.calc_variance(data_x)
		w0 = self.calc_mean(data_y) - (w1 * self.calc_mean(data_x))
		return w0, w1

	def linear_regression(self, data_x, data_y):
		data_x_mean = self.calc_mean(data_x)
		data_y_mean = self.calc_mean(data_y)

		data_x_variance = self.calc_variance(data_x)
		data_y_variance = self.calc_variance(data_y)

		w0, w1 = self.calc_linear_regression_coefficients(data_x, data_y)
		return w0, w1

	def encode(self, sample):
		encoding = np.zeros(shape=(self.num_traits,))
		for key, val in sample.items():
			# Get location of the trait
			if key in self.trait_locations:
				trait_location = self.trait_locations[key]
				if bool(self.traits[trait_location]):
					if val in self.traits[trait_location]:
						encoding[trait_location] = self.traits[trait_location][val]
				else:
					encoding[trait_location] = val
		return encoding

	@staticmethod
	def normalize_and_scale(data):
		simple_scale = (data - data.min()) / (data.max() - data.min())
		data_mean = np.mean(simple_scale, axis=0)
		data_minus_mean = simple_scale - data_mean
		data_std = np.std(data_minus_mean, axis=0)
		return data_minus_mean / data_std

	def setup(self, samples):
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
			if self.samples_answers is None:
				self.samples_answers = np.asarray([self.positive_examples[i]["Original"][1]])
			else:
				self.samples_answers = np.vstack((self.samples_answers,
				                                  np.asarray([self.positive_examples[i]["Original"][1]])))

			if self.samples_encoded is None:
				self.samples_encoded = self.positive_examples[i]["Encoded"]
			else:
				self.samples_encoded = np.vstack((self.samples_encoded,
				                                  self.positive_examples[i]["Encoded"]))
			if self.positive_examples_encoded is None:
				self.positive_examples_encoded = self.positive_examples[i]["Encoded"]
				self.samples_answers = np.asarray([self.positive_examples[i]["Original"][1]])
			else:
				self.positive_examples_encoded = np.vstack((self.positive_examples_encoded,
				                                            self.positive_examples[i]["Encoded"]))

		for i in range(len(self.negative_examples)):
			self.negative_examples[i]["Encoded"] = self.encode(self.negative_examples[i]["Original"][0])
			if self.samples_answers is None:
				self.samples_answers = np.asarray([self.negative_examples[i]["Original"][1]])
			else:
				self.samples_answers = np.vstack((self.samples_answers,
				                                  np.asarray([self.negative_examples[i]["Original"][1]])))
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
		return

	def solve(self, samples, new_monster):
		# Add your code here!
		#
		# The first parameter to this method will be a labeled list of samples in the form of
		# a list of 2-tuples. The first item in each 2-tuple will be a dictionary representing
		# the parameters of a particular monster. The second item in each 2-tuple will be a
		# boolean indicating whether this is an example of this species or not.
		#
		# The second parameter will be a dictionary representing a newly observed monster.
		#
		# Your function should return True or False as a guess as to whether or not this new
		# monster is an instance of the same species as that represented by the list.
		self.setup(samples=samples)
		self.positive_examples_encoded_scaled = self.normalize_and_scale(self.positive_examples_encoded)
		self.negative_examples_encoded_scaled = self.normalize_and_scale(self.negative_examples_encoded)
		monsters = [i[0] for i in samples]
		results = [i[1] for i in samples]

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
		# self.avg_positive[i] = sum(self.positive_examples_encoded[:, i]) / len(self.positive_examples_encoded[:, i])
		# self.avg_negative[i] = sum(self.negative_examples_encoded[:, i]) / len(self.negative_examples_encoded[:, i])
		self.avg_positive = np.mean(self.positive_examples_encoded, axis=0)
		self.avg_negative = np.mean(self.positive_examples_encoded, axis=0)
		self.positive_examples_std = np.std(self.positive_examples_encoded, axis=0)
		self.negative_examples_std = np.std(self.negative_examples_encoded, axis=0)
		self.samples_encoded_scaled = self.normalize_and_scale(self.samples_encoded)
		self.samples_answers_encoded = np.zeros(shape=(len(self.samples_answers),))
		self.samples_answers_encoded[:] = self.samples_answers[:, 0]
		multiplier = np.asarray([0.10014, 0.11766, 0.13479, 0.12895, 0.12132, 0.09576,
		                         0.12396, 0.05205, 0.02682, 0.02743, 0.03000, 0.04114])
		# 1 STD = 68.3%
		# 2 STD = 95.4%
		# 3 STD = 99.7%
		possible_answers = {"Positive", "Negative"}
		result = np.zeros(shape=(12,))
		new_monster_encoded = self.encode(new_monster)
		outside_one_std_negative = 0
		outside_one_std_positive = 0
		outside_two_std_negative = 0
		outside_two_std_positive = 0
		inside_one_std_negative = 0
		inside_one_std_positive = 0
		inside_two_std_negative = 0
		inside_two_std_positive = 0

		for key, val in new_monster.items():
			idx = int(self.trait_locations[key])
			positive_one_std_min, positive_one_std_max = self.avg_positive[idx] - self.positive_examples_std[idx], \
			                                             self.avg_positive[idx] + self.positive_examples_std[idx]
			negative_one_std_min, negative_one_std_max = self.avg_negative[idx] - self.negative_examples_std[idx], \
			                                             self.avg_negative[idx] + self.negative_examples_std[idx]
			positive_two_std_min, positive_two_std_max = self.avg_positive[idx] - (2 * self.positive_examples_std[idx]), \
			                                             self.avg_positive[idx] + (2 * self.positive_examples_std[idx])
			negative_two_std_min, negative_two_std_max = self.avg_negative[idx] - (2 * self.negative_examples_std[idx]), \
			                                             self.avg_negative[idx] + (2 * self.negative_examples_std[idx])
			if key in {"size", "color", "covering", "foot-type", "leg-count", "arm-count", "eye-count"}:
				if new_monster_encoded[idx] < negative_one_std_min:
					outside_one_std_negative += 1
					if new_monster_encoded[idx] < negative_two_std_min:
						outside_two_std_negative += 1
						if "Negative" in possible_answers:
							possible_answers.remove("Negative")
					else:
						inside_two_std_negative += 1
				else:
					inside_one_std_negative += 1

				if new_monster_encoded[idx] > negative_one_std_max:
					outside_one_std_negative += 1
					if new_monster_encoded[idx] > negative_two_std_max:
						outside_two_std_negative += 1
						if "Negative" in possible_answers:
							possible_answers.remove("Negative")
					else:
						inside_two_std_negative += 1
				else:
					inside_one_std_negative += 1

				if new_monster_encoded[idx] < positive_one_std_min:
					outside_one_std_positive += 1
					if new_monster_encoded[idx] < positive_two_std_min:
						outside_two_std_positive += 1
						if "Positive" in possible_answers:
							possible_answers.remove("Positive")
					else:
						inside_two_std_positive += 1
				else:
					inside_one_std_positive += 1

				if new_monster_encoded[idx] > positive_one_std_max:
					outside_one_std_positive += 1
					if new_monster_encoded[idx] > positive_two_std_max:
						outside_two_std_positive += 1
						if "Positive" in possible_answers:
							possible_answers.remove("Positive")
					else:
						inside_two_std_positive += 1
				else:
					inside_one_std_positive += 1

		if len(possible_answers) == 1:
			if possible_answers.pop() == "Positive":
				return True
			else:
				return False
		elif len(possible_answers) == 0:
			return True
		elif len(possible_answers) == 2:
			if new_monster["leg-count"] % 2 == 1 or new_monster["eye-count"] % 2 == 1 or new_monster[
				"eye-count"] == 1 or 2 < new_monster["eye-count"] < 8 or new_monster["arm-count"] % 2 == 1:
				return True
			else:
				return False
		else:
			return True
