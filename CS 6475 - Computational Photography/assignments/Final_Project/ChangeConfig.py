import os
import sys
import argparse
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from xml.dom import minidom


WEBCONFIG_PATH = "C:/DEV/Pharmacy/DEV/QS1.Pharmacy/QS1.Framework.ServiceHost/Web.config"
APPCONFIG_PATH1 = "C:/DEV/Pharmacy/DEV/QS1.Pharmacy/QS1.Framework.TaskSchedulerService/App.config"
APPCONFIG_PATH2 = "C:/DEV/Pharmacy/DEV/QS1.Pharmacy/QS1.Framework.ServiceBusService/App.config"

SHORTSPRINTS = {"IP": "10.0.200.1", "CustomerNumber": "8A12"}
LONGSPRINTS = {"IP": "10.0.200.216", "CustomerNumber": "8A11"}


def ShortRoutine(num):
	# Customer Number should be 8A12
	if num != SHORTSPRINTS["CustomerNumber"]:
		print("Incorrect Customer Number for Short Sprints")
		return
	print()


def LongRoutine():
	# Customer number should be
	print()


def DefaultRoutine(num):
	# Customer Number should be 8A12 or 9997
	if num == "9997" or num == "8A20":
		ChangeWebConfig()
	print("Incorrect Customer number to use the defaults")
	return


def ChangeWebConfig():
	file = open(WEBCONFIG_PATH, mode='r')
	file = file.readlines()
	for i in file:
		print(i)
	print()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Change the Config files')
	parser.add_argument('customerNumber', type=str, help='Customer Number')
	arguments = parser.parse_args()
	if arguments.customerNumber == "8A12":
		# We want to use Short Sprints
		print("You input {}, so that will be changed in the following files:\n{}\n{}\n{}\n".format(
			arguments.customerNumber, WEBCONFIG_PATH, APPCONFIG_PATH1, APPCONFIG_PATH2))
		ShortRoutine(arguments.customerNumber)
		
	if arguments.customerNumber == "8A11":
		# We want to use Short Sprints
		print("You input {}, so that will be changed in the following files:\n{}\n{}\n{}\n".format(
			arguments.customerNumber, WEBCONFIG_PATH, APPCONFIG_PATH1, APPCONFIG_PATH2))
		LongRoutine(arguments.customerNumber)

	elif arguments.customerNumber == "8A20":
		# We want to use My local environment 9997 is typical but mine is 8A20
		print("You input {}, so that will be changed in the following files:\n{}\n{}\n{}\n".format(
			arguments.customerNumber, WEBCONFIG_PATH, APPCONFIG_PATH1, APPCONFIG_PATH2))
		DefaultRoutine(arguments.customerNumber)

	elif arguments.customerNumber == "9997":
		# We want to use normal 9997
		print("You input {}, so that will be changed in the following files:\n{}\n{}\n{}\n".format(
			arguments.customerNumber, WEBCONFIG_PATH, APPCONFIG_PATH1, APPCONFIG_PATH2))
		DefaultRoutine(arguments.customerNumber)

	else:
		print("You input {}, so that will be changed in the following files:\n{}\n{}\n{}\n".format(
			arguments.customerNumber, WEBCONFIG_PATH, APPCONFIG_PATH1, APPCONFIG_PATH2))

	print()
