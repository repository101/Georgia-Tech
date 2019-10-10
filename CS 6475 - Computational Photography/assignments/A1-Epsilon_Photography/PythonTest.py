import urllib.request
import json
import sys
import traceback
import numpy as np


def validate_input(input):
	"""
	A helper function to validate command line argument
	:param input:
	:return: True or False
	"""
	valid_text = ['loc', 'location', 'pass', 'people']
	lower_input = input.lower()
	if lower_input not in valid_text:
		return False
	else:
		return True


def vectorized_print(a):
	"""
	A Vectorized version of print to remove the need to loop through list
	:param a: a NumPy array of objects
	:return:
	"""
	print("Name: ", a)
	return


if __name__ == '__main__':
	try:
		# A dictionary to hold to specific URLs for the various command line arguments
		# 	I specified the Longitude/Latitude to be Charlotte, NC for the 'pass' argument
		url_dict = {"loc": "http://api.open-notify.org/iss-now.json",
					"pass": "http://api.open-notify.org/iss-pass.json?lat=35.2271&lon=-80.8431",
					"people": "http://api.open-notify.org/astros.json"}

		# Partial validate of command line argument, if no argument is passed or if too many
		if len(sys.argv) <= 1:
			print("A command line argument is required")
			print("Script will be closing now as a command line argument was required")
			sys.exit(0)

		if len(sys.argv) > 2:
			print("You are using too many arguments")
			print("Script will be closing now as only a single command line argument was expected")
			sys.exit(0)

		# Command Line Argument
		argument = sys.argv[1]

		# Validate the specified argument
		if not validate_input(argument):
			print("Argument is not valid, please try again")
			print("Script will be closing now as a non-valid command line argument was received")
			sys.exit(0)

		else:
			# Open the URL and read the data for processing
			response = urllib.request.urlopen(url_dict[argument])
			response_data = json.loads(response.read())

			if argument == 'loc':
				print("The ISS current location at {} is {{ \"longitude\" : {}, \"latitude\" : {} }}".format(
					response_data['timestamp'],
					response_data['iss_position']['longitude'],
					response_data['iss_position']['latitude']))

			elif argument == 'pass':
				# I was not sure the number of passes ISS could have, so I stop at 5
				print("Returning up to the first 5 results")
				print("The ISS will be overhead {{ \"longitude\" : {}, \"latitude\" : {} }} for {} passes".format(
					response_data['request']['longitude'],
					response_data['request']['latitude'],
					response_data['request']['passes']))
				print("The duration and time of those passes are: ")

				for pass_count in range(len(response_data['response'])):
					# Check number of passes and leave loop if we exceed limit
					if pass_count >= 5:
						continue
					print("Pass# {}".format(pass_count))
					print("Duration: {}\nTime: {}\n".format(
						response_data['response'][pass_count]['duration'],
						response_data['response'][pass_count]['risetime']))

			elif argument == 'people':

				print("Displaying up to the first 10 results\n")
				people_list = response_data['people']

				vfunc = np.vectorize(vectorized_print, doc='Just to print using the array, faster than loop', otypes=[object])
				bins = np.unique(np.asarray([data['craft'] for data in people_list], dtype=object))
				bin_index = [i for i in range(len(bins))]
				values = np.asarray([data for data in response_data['people']], dtype=object)
				craft_index = np.asarray([np.argwhere(bins == data['craft'])[0][0] for data in people_list], dtype=object)
				for i in range(len(bins)):
					print("Craft: ", bins[i])
					t = np.argwhere(craft_index == i)
					p = [people_list[l[0]]['name'] for l in t]
					vfunc(p)
					print()

			else:
				print("An error occurred")
				raise ValueError("The argument was not an expected value")

		print("\nFinished")

	except BaseException:
		print(sys.exc_info()[0])
		print(traceback.format_exc())
	finally:
		print("Press Enter to continue ...")
		input()
