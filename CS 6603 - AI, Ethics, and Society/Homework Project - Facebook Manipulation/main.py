import os
import sys
import time

import numpy as np
import pandas as pd

if __name__ == "__main__":
	filePathInteract = "./ads_information/advertisers_you've_interacted_with.json"
	filePathContact = "./ads_information/advertisers_who_uploaded_a_contact_list_with_your_information.json"
	
	interactDF = pd.read_json(filePathInteract)
	contactDF = pd.read_json(filePathContact)
	print()
