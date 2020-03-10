import os
import pathlib
import sys
import numpy as np
import pandas as pd

from Boosting import Boosting
from Decision_Tree import DecisionTree
from K_NearestNeighbors import KNN
from LoadData import LoadData
from NeuralNetwork import NeuralNetwork
from SupportVectorMachine import SVM

TESTING = False
DECISION_TREE = False
SUPPORT_VECTOR = False
NEURAL_NET = True
K_NEAREST = False
BOOSTING = False
NORMALIZE_DATA = True
GENERATE_REPORTS = False


if __name__ == "__main__":
    try:
        cwd = pathlib.Path().absolute()
        training_data_path = "{}/mnist-train-data.csv".format(cwd)
        testing_data_path = "{}/mnist-test-data.csv".format(cwd)
        DataSetName = "MNIST"
        dataset_list = ["MNIST", "Fashion_MNIST"]
        if TESTING:
            training_data_path = "{}/sub-sample-mnist-train-data.csv".format(cwd)
            training_labels, training_data, _ = LoadData(training_data_path, normalize=NORMALIZE_DATA)
            if DECISION_TREE:
                DecisionTreeClassifier = DecisionTree(train_data=training_data, train_labels=training_labels,
                                                      split_method='entropy', tree_max_depth=50,
                                                      dataset_name=DataSetName, gen_reports=GENERATE_REPORTS)
            if SUPPORT_VECTOR:
                print("Initializing Support Vector Classifier")
                SupportVectorClassifier = SVM(train_data=training_data, train_labels=training_labels,
                                              dataset_name=DataSetName, gen_reports=GENERATE_REPORTS)
            
            if NEURAL_NET:
                print("Initializing Neural Network Classifier")
                NeuralNetworkClassifier = NeuralNetwork(train_data=training_data, train_labels=training_labels,
                                                        dataset_name=DataSetName, gen_reports=GENERATE_REPORTS)
            
            if K_NEAREST:
                print("Initializing K-Nearest Neighbor Classifier")
                KNNClassifier = KNN(train_data=training_data, train_labels=training_labels,
                                    dataset_name=DataSetName, gen_reports=GENERATE_REPORTS)
            
            if BOOSTING:
                print("Initializing Boosted Decision Tree Classifier")
                BoostedClassifier = Boosting(train_data=training_data, train_labels=training_labels,
                                             dataset_name=DataSetName, gen_reports=GENERATE_REPORTS)
        
        if not TESTING:
            for i in dataset_list:
                if i == "MNIST":
                    training_data_path = "{}/mnist-train-data.csv".format(cwd)
                    testing_data_path = "{}/mnist-test-data.csv".format(cwd)
                else:
                    training_data_path = "{}/fashion-mnist-train-data.csv".format(cwd)
                    testing_data_path = "{}/fashion-mnist-test-data.csv".format(cwd)
                training_labels, training_data, _ = LoadData(training_data_path, normalize=NORMALIZE_DATA)
                testing_labels, testing_data, _ = LoadData(testing_data_path, normalize=NORMALIZE_DATA)

                print("Beginning Decision Tree Segment for {} Dataset \n".format(i))
                DecisionTreeClassifier = DecisionTree(train_data=training_data, train_labels=training_labels,
                                                      test_data=testing_data, test_labels=testing_labels,
                                                      split_method='entropy', tree_max_depth=100,
                                                      dataset_name=i, gen_reports=GENERATE_REPORTS)
            
                print("Beginning Support Vector Segment for {} Dataset \n".format(i))
                SupportVectorClassifier = SVM(train_data=training_data, train_labels=training_labels,
                                              test_data=testing_data, test_labels=testing_labels,
                                              dataset_name=i, gen_reports=GENERATE_REPORTS)
            
                print("Beginning Neural Network Segment for {} Dataset \n".format(i))
                NeuralNetworkClassifier = NeuralNetwork(train_data=training_data, train_labels=training_labels,
                                                        test_data=testing_data, test_labels=testing_labels,
                                                        dataset_name=i, gen_reports=GENERATE_REPORTS)
            
                print("Beginning K Nearest Neighbor Segment for {} Dataset \n".format(i))
                KNNClassifier = KNN(train_data=training_data, train_labels=training_labels,
                                    test_data=testing_data, test_labels=testing_labels,
                                    dataset_name=i, gen_reports=GENERATE_REPORTS)
                       
                print("Beginning K Nearest Neighbor Segment for {} Dataset \n".format(i))
                BoostedClassifier = Boosting(train_data=training_data, train_labels=training_labels,
                                             test_data=testing_data, test_labels=testing_labels,
                                             dataset_name=i, gen_reports=GENERATE_REPORTS)
                
        print("OMG IT FINISHED... I am as surprised as you are")
        sys.exit()
    except Exception as MainException:
        print("Exception occurred while executing the Main function: \n", MainException)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
