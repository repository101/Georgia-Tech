import os
import sys

from Training_Set_Compare import BoostedDecisionTreeDepthCompare, BoostedDecisionTreeTrainingSizeCompare, \
    DecisionTreeDepthCompare, DecisionTreeTrainingSizeCompare, KNNTrainingSizeCompare, NeuralNetworkTrainingSizeCompare, \
    SVMTrainingSizeCompare


def Generate_Report(training_data, training_labels, testing_data, testing_labels,
                    result_csv_path=None, classifier_type='tree', dataset="MNIST"):
    try:
        if classifier_type.lower() == 'tree':
            DecisionTreeTrainingSizeCompare(training_data, training_labels,
                                            testing_data, testing_labels, dataset=dataset)
        elif classifier_type.lower() == 'svm':
            SVMTrainingSizeCompare(training_data, training_labels, testing_data, testing_labels, dataset=dataset)

        elif classifier_type.lower() == 'neural':
            NeuralNetworkTrainingSizeCompare(training_data, training_labels,
                                             testing_data, testing_labels, dataset=dataset)
        
        elif classifier_type.lower() == "knn":
            KNNTrainingSizeCompare(training_data, training_labels, testing_data, testing_labels, dataset=dataset)
            
        if classifier_type.lower() == "tree":
            DecisionTreeDepthCompare(training_data, training_labels, testing_data, testing_labels, dataset=dataset)

        if classifier_type.lower() == "boost":
            BoostedDecisionTreeTrainingSizeCompare(training_data, training_labels, testing_data, testing_labels,
                                                   dataset=dataset)
            BoostedDecisionTreeDepthCompare(training_data, training_labels, testing_data, testing_labels,
                                            dataset=dataset)
            
    except Exception as Generate_Report_Exception:
        print("Exception occurred while attempting to generate examples for report. \n", Generate_Report_Exception)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
