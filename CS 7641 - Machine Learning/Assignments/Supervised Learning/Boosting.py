import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import parallel_backend

from Generate_Report import Generate_Report


class Boosting:
    def __init__(self, train_data=None, train_labels=None, test_data=None, test_labels=None, split_value=0.7,
                 n_estimators=5, verbose=True, useGridSearch=False, learning_rate=1., verbose_int=3,
                 cv_folds=3, useGraphViz=False, boostType='gradient', max_depth=3, subsample=0.5,
                 max_iteration=10, use_pca=False, dataset_name="MNIST", gen_reports=False):
        self.ClassifierName = "Boosted Decision Tree"
        self.DataSetName = dataset_name
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.use_pca = use_pca
        self.gen_reports = gen_reports
        self.pca_applied = False
        self.train_test_split_percentage = split_value
        self.n_estimators = n_estimators
        self.boostType = boostType
        self.max_iterations = max_iteration
        self.subsample = subsample
        self.max_depth = max_depth
        self.cv_folds = cv_folds
        self.useGraphViz = useGraphViz
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.verbose_int = verbose_int
        self.useGridSearch = useGridSearch
        self.train_test_split_percentage = split_value
        self.OnInitialize()
        self.Training_Complete = False
        self.Result_CSV_path = None
        self.Classifier = None
        self.GridSearchParametersForADA = {'base_estimator': [self.Classifier],
                                           'n_estimator': [i for i in range(0, 600, 50)],
                                           'learning_rate': [0.1, 0.5, 1.0, 1.5, 2.0]}
        
        self.GridSearchParametersForGradient = {'base_estimator': [self.Classifier],
                                                'n_estimator': [i for i in range(0, 600, 50)],
                                                'learning_rate': [0.1, 0.5, 1.0, 1.5, 2.0]}
    
    def OnInitialize(self):
        try:
            if self.useGraphViz:
                os.environ["PATH"] += os.pathsep + 'C:/ProgramData/Anaconda3/envs/CS7641/Library/bin/graphviz/'

            if self.test_data is None and self.test_labels is None:
                self.TrainTestSplit()
                
            Scaler = StandardScaler().fit(self.train_data)

            self.train_data = Scaler.transform(self.train_data)
            self.test_data = Scaler.transform(self.test_data)

            if self.gen_reports:
                Generate_Report(training_data=self.train_data, training_labels=self.train_labels,
                                testing_data=self.test_data, testing_labels=self.test_labels, classifier_type='boost',
                                dataset=self.DataSetName)

            if (self.train_data is not None) and (self.train_labels is not None):
                self.TrainClassifier()
            
            if self.Training_Complete:
                self.TestClassifier()
        except Exception as OnInitializeException:
            print("Exception occurred during the Initialization of the Boosting object. \n", OnInitializeException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    
    def TrainTestSplit(self):
        try:
            if self.train_data is None or self.train_labels is None:
                raise Exception
            x_train, x_test, y_train, y_test = train_test_split(self.train_data, self.train_labels,
                                                                train_size=self.train_test_split_percentage)
            
            if self.use_pca:
                scaler = StandardScaler()
                scaler.fit(x_train)
                x_train = scaler.transform(x_train)
                x_test = scaler.transform(x_test)
            
            self.train_data = x_train
            self.train_labels = y_train
            self.test_data = x_test
            self.test_labels = y_test
        except Exception as TrainTestSplitException:
            print("Exception occurred while splitting the data into a training and testing set. \n",
                  TrainTestSplitException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    
    def TrainClassifier(self):
        try:
            print("Beginning to train the classifier\n")
            baseDecisionTree = DecisionTreeClassifier(max_depth=self.max_depth)
            with parallel_backend('threading'):
                if self.boostType.lower() == 'ada':
                    self.Classifier = AdaBoostClassifier(base_estimator=baseDecisionTree,
                                                         n_estimators=self.n_estimators,
                                                         learning_rate=self.learning_rate)
                    print("Finished initializing ADABoostClassifier")
                
                elif self.boostType.lower() == 'gradient':
                    self.Classifier = GradientBoostingClassifier(n_estimators=self.n_estimators,
                                                                 max_depth=self.max_depth,
                                                                 verbose=self.verbose)
                    print("Finished initializing GradientBoostClassifier")
                
                self.Classifier.fit(self.train_data, self.train_labels)

            print("Finished Generating Charts for Report")
            print("Training of the classifier has finished\n")
            self.Training_Complete = True
        except Exception as TrainClassifierException:
            print("Exception occurred while training the classifier. \n", TrainClassifierException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    
    def TestClassifier(self):
        try:
            print("Beginning to test the classifier\n")
            y_pred = self.Classifier.predict(self.test_data)
            temp_confusion_matrix = confusion_matrix(self.test_labels, y_pred)
            temp_classification_report = classification_report(self.test_labels, y_pred)
            print("Confusion Matrix \n {} \n".format(temp_confusion_matrix))
            print("Classification Report \n {} \n".format(temp_classification_report))
            disp = plot_confusion_matrix(self.Classifier, self.test_data, self.test_labels, values_format=".4g")
            disp.figure_.suptitle("Confusion Matrix")
            print("TESTING")
            print(disp.confusion_matrix)
            
            plt.savefig("Boosting_Results/Boosting_Images/Base_Boosting"
                        "_{}_ConfusionMatrix_{}.png".format(self.DataSetName, datetime.now().strftime("%m-%d~%I%M %p")))
            
            if self.useGridSearch:
                self.RunGridSearch()
            print("Testing of the classifier has finished\n")
        except Exception as TestClassifierException:
            print("Exception occurred while testing the classifier. \n", TestClassifierException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    
    def RunGridSearch(self):
        try:
            if self.train_data.shape[0] > 5000 and self.ClassifierName.lower() != "decision tree":
                print("Sorry dataset is too large to run GridSearch and/or Cross-Validation.\n")
                return
            print("Starting Grid Search")
            with parallel_backend('threading'):
                if self.boostType.lower() == 'ada':
                    scoring_clf = GridSearchCV(AdaBoostClassifier(), param_grid=self.GridSearchParametersForADA,
                                               scoring='accuracy', verbose=self.verbose_int, cv=self.cv_folds)
                else:
                    scoring_clf = GridSearchCV(GradientBoostingClassifier(),
                                               param_grid=self.GridSearchParametersForGradient,
                                               scoring='accuracy', verbose=self.verbose_int, cv=self.cv_folds)
                
                scoring_clf.fit(self.train_data, self.train_labels)
            cross_validation_results = pd.DataFrame(scoring_clf.cv_results_)
            self.Result_CSV_path = "Boosting_Results/BoostedDecisionTree_" \
                                   "{}_GridSearch_Results_{}.csv".format(self.DataSetName, datetime.
                                                                            now().strftime("%m-%d~%I%M %p"))
            cross_validation_results.to_csv(self.Result_CSV_path)
            print("Grid Search has ended.\n")
        
        except Exception as RunGridSearchException:
            print("Exception occurred while running GridSearch. \n", RunGridSearchException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
