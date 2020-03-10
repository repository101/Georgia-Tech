import os
import pathlib
import sys
import numpy as np
import pandas as pd

from LoadData import LoadData


TESTING = False
DECISION_TREE = False
SUPPORT_VECTOR = False
NEURAL_NET = True
K_NEAREST = False
BOOSTING = False
NORMALIZE_DATA = False
GENERATE_REPORTS = False

if __name__ == "__main__":
    try:
        cwd = pathlib.Path().absolute()
        training_data_path = "{}/sub-sample-mnist-train-data.csv".format(cwd)
        testing_data_path = "{}/sub-sample-mnist-test-data.csv".format(cwd)
        DataSetName = "MNIST"
        training_labels, training_data, _ = LoadData(training_data_path, normalize=NORMALIZE_DATA)
        temp_list = ["Labels"]
        col_list = [str(i) for i in range((28*28))]
        for i in col_list:
            temp_list.append(i)
        d = {"Labels":training_labels, }
        df = pd.DataFrame(data=training_data)
        df.insert(0, "Labels", training_labels)
        print(df.dtypes)
        df.to_csv("test.tsv", sep="\t")
        print("OMG IT FINISHED... I am as surprised as you are")
        sys.exit()
    except Exception as MainException:
        print("Exception occurred while executing the Main function: \n", MainException)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
