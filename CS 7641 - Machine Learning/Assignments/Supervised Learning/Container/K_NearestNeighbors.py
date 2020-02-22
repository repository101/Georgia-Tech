import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import parallel_backend

from Generate_Report import Generate_Report


class KNN:
    def __init__(self, train_data=None, train_labels=None, test_data=None, test_labels=None, split_value=0.7,
                 weights='uniform', algorithm='auto', number_neighbors=5, verbose=True, useGridSearch=False,
                 n_jobs=1, verbose_int=3, cv_folds=5, useGraphViz=False, use_pca=False, dataset_name="MNIST",
                 gen_reports=False):
        self.ClassifierName = "K-Nearest Neighbors"
        self.DataSetName = dataset_name
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.use_pca = use_pca
        self.pca_applied = False
        self.train_test_split_percentage = split_value
        self.weights = weights
        self.gen_reports = gen_reports
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.number_neighbors = number_neighbors
        self.verbose = verbose
        self.verbose_int = verbose_int
        self.useGraphViz = useGraphViz
        self.useGridSearch = useGridSearch
        self.GridSearchParameters = {'n_neighbors': [i for i in range(1, 30, 2)],
                                     'weights': ['uniform', 'distance'],
                                     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
        self.train_test_split_percentage = split_value
        self.OnInitialize()
        self.Result_CSV_path = None
        self.Training_Complete = False
        self.Classifier = None
    
    def OnInitialize(self):
        try:
            if self.useGraphViz:
                os.environ["PATH"] += os.pathsep + 'C:/ProgramData/Anaconda3/envs/CS7641/Library/bin/graphviz/'
            if self.test_data is None and self.test_labels is None:
                self.TrainTestSplit()
            
            self.Classifier = KNeighborsClassifier(n_neighbors=self.number_neighbors, algorithm=self.algorithm)

            if self.gen_reports:
                Generate_Report(training_data=self.train_data, training_labels=self.train_labels,
                                testing_data=self.test_data, testing_labels=self.test_labels, classifier_type='knn',
                                dataset=self.DataSetName)
            
            if (self.train_data is not None) and (self.train_labels is not None) and (self.Classifier is not None):
                self.TrainClassifier()
            
            if self.Training_Complete:
                self.TestClassifier()
        except Exception as OnInitializeException:
            print("Exception occurred during the Initialization of the K-Nearest Neighbor object. \n",
                  OnInitializeException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    
    def TrainTestSplit(self):
        try:
            if self.train_data is None or self.train_labels is None:
                raise Exception
            x_train, x_test, y_train, y_test = train_test_split(self.train_data,
                                                                self.train_labels,
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
            with parallel_backend('threading'):
                self.Classifier.fit(self.train_data, self.train_labels)
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
            print("Test set score: %f" % self.Classifier.score(self.test_data, self.test_labels))
            y_pred = self.Classifier.predict(self.test_data)
            
            classification_report = metrics.classification_report(self.test_labels, y_pred=y_pred)
            print("Classification Report\n {} \n".format(classification_report))
            confusion_matrix = metrics.plot_confusion_matrix(self.Classifier, self.test_data,
                                                             self.test_labels, values_format=".4g")
            confusion_matrix.figure_.suptitle("Confusion Matrix")
            print("Confusion matrix:\n%s" % confusion_matrix.confusion_matrix)
            plt.savefig("KNN_Results/K_Nearest_Neighbor_Images"
                        "/Base_Boosting_{}_ConfusionMatrix_{}.png"
                        .format(self.DataSetName, datetime.now().strftime("%m-%d~%I%M %p")))
            
            model_acc = metrics.precision_score(self.test_labels, y_pred, average='weighted')
            test_acc = metrics.accuracy_score(self.test_labels, y_pred)
            conf_mat = metrics.confusion_matrix(self.test_labels, y_pred)
            print('\nKNN Trained Classifier Accuracy: ', model_acc)
            print('\nPredicted Values: ', y_pred[:10])
            print('\nAccuracy of Classifier on Validation Images: ', test_acc)
            print('\nConfusion Matrix: \n', conf_mat)
            
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
                knn_clf = GridSearchCV(KNeighborsClassifier(), param_grid=self.GridSearchParameters,
                                       scoring='accuracy', verbose=self.verbose_int, cv=self.cv_folds)
                knn_clf.fit(self.train_data, self.train_labels)
            cross_validation_results = pd.DataFrame(knn_clf.cv_results_)
            
            self.Result_CSV_path = "KNN_Results/K_NearestNeighbors_" \
                                   "{}_GridSearch_Results_{}.csv".format(self.DataSetName, datetime.
                                                                            now().strftime("%m-%d~%I%M %p"))
            cross_validation_results.to_csv(self.Result_CSV_path)
            print("Grid Search has ended.\n")
        
        except Exception as RunGridSearchException:
            print("Exception occurred while running GridSearch. \n", RunGridSearchException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
