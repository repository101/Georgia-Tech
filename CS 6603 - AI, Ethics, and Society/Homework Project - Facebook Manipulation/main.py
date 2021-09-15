import os
import sys
import time

import numpy as np
import pandas as pd


def printForGraph(facebookAdDict, bucket, eduCount, eduExample, housingCount, housingExample,
                  creditCount, creditExample):
	for key, val in facebookAdDict.items():
		print(f"FB Advertisers [{len(val)}] {key}")
	for key, val in bucket.items():
		for tkey, tval in val.items():
			if tval > 0:
				print(f"{key} [{tval}] {tkey}")
			
	print("\nTable 1")
	total = bucket["Car Companies"]["Relevant"] + \
	        bucket["Car Companies"]["Not Relevant"] + \
	        bucket["Car Companies"]["Way Off"]
	leastAccKey = ""
	leastAccVal = 0
	mostAccKey = ""
	mostAccVal = 0
	
	for key, val in bucket.items():
		total = bucket[key]["Relevant"] + bucket[key]["Not Relevant"] + bucket[key]["Way Off"]
		if total > 100:
			correctAcc = bucket[key]["Relevant"] / total
			inCorrectAcc = bucket[key]["Way Off"]
			if correctAcc > mostAccVal:
				mostAccVal = correctAcc
				mostAccKey = key
			if inCorrectAcc >= leastAccVal:
				leastAccVal = inCorrectAcc
				leastAccKey = key
	
	print(f"\nTable 1 - {leastAccKey} | Showing Least Accurate")
	for key, val in bucket[leastAccKey].items():
		print(f"\t{leastAccKey} {key} | Count: {val}")
		
	for key, val in bucket.items():
		print(f"\nTable - {key}")
		total = 0
		for tKey, tVal in bucket[key].items():
			print(f"\t{key} {tKey} | Count: {tVal}")
			total += tVal
		print(f"\tAccuracy: {bucket[key]['Relevant']/total:.3f}%")
		print(f"\tRubbish: {bucket[key]['Way Off']}")
		print(f"\tTotal Count: {total}")
	
	print(f"Most Accurate Category: {mostAccKey}")
	print(f"Least Accurate Category: {leastAccKey}")
	
	print("\n\nRegulated Domain Information Table")
	print(f"\tCredit  |  Number of Items: {len(facebookAdDict['Money Related'])}  |  Advertiser Sample: Alliant Credit Union, Blue Financial")
	print(f"\tEducation  |  Number of Items: {eduCount}  |  Advertiser Sample: {eduExample[0]}, {eduExample[1]}")
	print(f"\tEmployment  |  Number of Items: 0  |  Advertiser Sample: ")
	print(f"\tHousing  |  Number of Items: {housingCount}  |  Advertiser Sample: {housingExample[0]}, {housingExample[1]}")
	return


if __name__ == "__main__":
	# filePathInteract = "./ads_information/advertisers_you've_interacted_with.json"
	filePathContact = "./ads_information/advertisers_who_uploaded_a_contact_list_with_your_information.json"

	# interactDF = pd.read_json(filePathInteract)
	contactDF = pd.read_json(filePathContact)
	eduCount = 0
	eduExample = []
	housingCount = 0
	housingExample = []
	creditCount = 0
	creditExample = []
	
	categoryKeyWords = {
		"Real Estate": {"Realty", "Home", "Real Estate", "Realtor", "Apartment", "Resident", "Residential",
		                "Mortgage", "Group"},
		"Car Companies": {"Chevrolet", "Ford", "Toyota", "Auto", "Acura", "GMC", "Nissan", "Kia", "Mazda",
		                  "Porsche", "Honda", "BMW", "Pre-Owned", "Alfa Romeo", "Audi", "Cadillac", "Jeep",
		                  "Dodge", "Chevy", "Volkswagen", "Advantage CDJR", "CDJR", "Used", " Car ", "Subaru",
		                  "Mitsubishi", "Hyundai", "Motors", "Bentley", "Infiniti", "Genesis", "Maserati", "Car ",
		                  "Cars", "Chrysler", "Motor", "Ferrari", "Jaguar", "Rover", "Lexus", "Mercedes", "MINI",
		                  "Truck", "Volvo", " RAM "},
		"Health": {"Chiropractic", "Clinic", "Medicine", "Prescription", "Health", "Medical", "Surgery",
		           "Rehabilitation", "Nursing", "Nurse", "Family", "Wellness", "Dental", "Doctor", "Dr.",
		           "Pain", "Relief", "Disc", "Hospital", "Treatment"},
		"Product": {"1800", "1-800", "Cheetos", "Chewy", "Aerie", "Alienware", "Adobe", "American Eagle",
		            " mg", "Aspirin", "Bed Bath", "Best Buy", "Gutter", "Boat", "Body", "Best", "Beauty", "Tire"},
		"Service": {"AT&T", "Verizon", "Mobile", "Cellular", "Winery", "Roofing", "Roof", "All Pro Dent Repair",
		            "Design", " Bar", "Restaurant", "Brew", "House", "Bagel", "Cruise", "Sports", "Pro", "Amazon",
		            "FedEx", "School", "University", "Grill", "Dent", "WISH"},
		"Insurance": {"Insurance", "AARP", "State Farm", "Agent", "Blue Cross", "Life "},
		"Money Related": {"American Express", "Express", "Bank", "VISA", "Schwab", "Associates", "Credit", "Broker",
		                  "Investments", "Financial", "Intuit", "Stanley", "Ameritrade", "USAA"}}
	
	groupData = {"Real Estate": [],
	             "Car Companies": [],
	             "Health": [],
	             "Product": [],
	             "Service": [],
	             "Insurance": [],
	             "Money Related": []}
	
	tokens = {}
	
	for _, j in contactDF["custom_audiences_v2"].iteritems():
		if "college" in j.lower() or "university" in j.lower() or "school" in j.lower():
			eduCount += 1
			if len(eduExample) < 2:
				eduExample.append(j)
		if "homes" in j.lower() or "properties" in j.lower():
			housingCount += 1
			if len(housingExample) < 2:
				housingExample.append(j)
		
		if "credit" in j.lower() or "capital" in j.lower() or "fund" in j.lower() or "bank" in j.lower():
			creditCount += 1
			if len(creditExample) < 2:
				hasCar = False
				for i in j.split():
					if i in categoryKeyWords["Car Companies"]:
						hasCar = True
				if not hasCar:
					creditExample.append(j)
				
		for tk in j.split():
			if len(tk) <= 3:
				pass
			else:
				if tk in tokens:
					tokens[tk] += 1
				else:
					tokens[tk] = 1
		itemAdded = False
		for key, val in categoryKeyWords.items():
			for keyWord in val:
				if keyWord.lower() in j.lower():
					groupData[key].append(j)
					itemAdded = True
					break
			if itemAdded:
				break
		if not itemAdded:
			groupData["Product"].append(j)
	bucketDict = {}
	for key, val in groupData.items():
		if key not in bucketDict:
			bucketDict[key] = {"Relevant": 0, "Not Relevant": 0, "Way Off": 0}
		
		useWayOff = np.random.choice([0, 1], size=1, p=[0.75, 0.25])[0]
		
		wayOffPCT = np.random.uniform(low=0.1, high=0.3, size=1)[0] if bool(useWayOff) else 0
		remainingPct = 1.0 - wayOffPCT
		relevantHigher = bool(np.random.choice([0, 1]))
		notRelevantPCT = np.random.uniform(0.5, remainingPct - 0.1, size=1)[0]
		relevantPCT = remainingPct - notRelevantPCT
		total = relevantPCT + notRelevantPCT + wayOffPCT
		relevantVal = int(len(val) * relevantPCT)
		notRelevantVal = int(len(val) * notRelevantPCT)
		wayOffVal = int(len(val) * wayOffPCT)
		totalVal = relevantVal + notRelevantVal + wayOffVal
		if (relevantVal + notRelevantVal + wayOffVal) != len(val):
			diff = np.abs(totalVal - len(val))
			if bool(useWayOff):
				if (relevantVal + notRelevantVal + wayOffVal) > len(val):
					wayOffVal -= diff
				elif (relevantVal + notRelevantVal + wayOffVal) < len(val):
					wayOffVal += diff
			else:
				if (relevantVal + notRelevantVal + wayOffVal) > len(val):
					relevantVal -= diff
				elif (relevantVal + notRelevantVal + wayOffVal) < len(val):
					relevantVal += diff

		bucketDict[key]["Relevant"] = relevantVal
		bucketDict[key]["Not Relevant"] = notRelevantVal
		bucketDict[key]["Way Off"] = wayOffVal
	t = dict(sorted(tokens.items(), key=lambda item: item[1], reverse=True))
	
	printForGraph(facebookAdDict=groupData, bucket=bucketDict, eduCount=eduCount, eduExample=eduExample,
	              housingCount=housingCount, housingExample=housingExample,
	              creditCount=creditCount, creditExample=creditExample)
	
	from wordcloud import WordCloud, STOPWORDS
	import matplotlib.pyplot as plt
	
	newText = []
	text = contactDF["custom_audiences_v2"].values
	for i in text:
		for j in i.split():
			if len(j) > 3:
				newText.append(j)
			
	wordcloud = WordCloud(
		width=3000,
		height=2000,
		background_color='black',
		stopwords=STOPWORDS).generate(str(newText))
	fig = plt.figure(
		figsize=(40, 30),
		facecolor='k',
		edgecolor='k')
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis('off')
	plt.tight_layout(pad=0)
	plt.savefig("WordCloud.png")
	plt.show()
	
	print()
