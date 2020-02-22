import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import pydotplus
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.tree._tree import TREE_LEAF
from sklearn.utils import parallel_backend

from Generate_Report import Generate_Report


class DecisionTree:
    def __init__(self, train_data=None, train_labels=None, test_data=None,
                 test_labels=None, split_value=0.7, split_method="entropy", tree_max_depth=50, useGridSearch=False,
                 verbose_int=3, n_jobs=1, cv_folds=2, useGraphViz=False, use_pca=False, dataset_name="MNIST",
                 gen_reports=False):
        self.ClassifierName = "Decision Tree"
        self.DataSetName = dataset_name
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.train_test_split_percentage = split_value
        self.split_method = split_method
        self.cv_folds = cv_folds
        self.gen_reports = gen_reports
        self.use_pca = use_pca
        self.pca_applied = False
        self.tree_max_depth = tree_max_depth
        self.useGridSearch = useGridSearch
        self.verbose_int = verbose_int
        self.n_jobs = n_jobs
        self.useGraphViz = useGraphViz
        self.GridSearchParameters = {'criterion': ['gini', 'entropy'],
                                     'min_samples_split': [i for i in range(4, 13, 4)],
                                     'min_samples_leaf': [i for i in range(4, 13, 4)],
                                     'max_depth': [i for i in range(5, 60, 5)]}
        self.OnInitialize()
        self.Training_Complete = False
        self.Result_CSV_path = None
        self.Classifier = None
    
    def OnInitialize(self):
        try:
            if self.useGraphViz:
                os.environ["PATH"] += os.pathsep + 'C:/ProgramData/Anaconda3/envs/CS7641/Library/bin/graphviz/'
            if self.test_data is None and self.test_labels is None:
                self.TrainTestSplit()
            
            if self.split_method.lower() == "gini":
                self.Classifier = DecisionTreeClassifier(criterion="gini", max_depth=self.tree_max_depth,
                                                         min_samples_leaf=6, min_samples_split=2)
            
            elif self.split_method.lower() == "entropy":
                self.Classifier = DecisionTreeClassifier(criterion="entropy", max_depth=self.tree_max_depth,
                                                         min_samples_leaf=6, min_samples_split=2)
            
            else:
                self.Classifier = DecisionTreeClassifier(criterion=self.split_method, max_depth=self.tree_max_depth)

            if self.gen_reports:
                Generate_Report(training_data=self.train_data, training_labels=self.train_labels,
                                testing_data=self.test_data, testing_labels=self.test_labels, classifier_type='tree',
                                dataset=self.DataSetName)
            
            if (self.train_data is not None) and (self.train_labels is not None) and (self.Classifier is not None):
                self.TrainClassifier()

                """
                Generate Image of Decision tree map prior to pruning
                """
                dot_data = StringIO()
                export_graphviz(self.Classifier, out_file=dot_data,
                                filled=True, rounded=True,
                                special_characters=True, class_names=[str(i) for i in range(10)])
                graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                graph.write_png('{}.png'.format("DecisionTree_Results//Decision_Tree_Images/Decision"
                                                "Tree_{}_Before_Pruning_{}"
                                                .format(self.DataSetName, datetime.now().strftime("%m-%d~%I%M %p"))))
                Image(graph.create_png())
                
                self.prune_index(0, 50)

                dot_data = StringIO()
                export_graphviz(self.Classifier, out_file=dot_data,
                                filled=True, rounded=True,
                                special_characters=True, class_names=[str(i) for i in range(10)])
                graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                graph.write_png('{}.png'.format("DecisionTree_Results/Decision_Tree_Images/Decision"
                                                "Tree_{}_After_Pruning_{}"
                                                .format(self.DataSetName, datetime.now().strftime("%m-%d~%I%M %p"))))
                Image(graph.create_png())
            
            if self.Training_Complete:
                self.TestClassifier()
        except Exception as OnInitializeException:
            print("Exception occurred during the Initialization of the Decision Tree object. \n", OnInitializeException)
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
            with parallel_backend('threading'):
                results = cross_val_score(estimator=self.Classifier,
                                          X=self.train_data, y=self.train_labels, cv=3,
                                          verbose=self.verbose_int)
            print(results)
            y_pred = self.Classifier.predict(self.test_data)
            temp_confusion_matrix = confusion_matrix(self.test_labels, y_pred)
            temp_classification_report = classification_report(self.test_labels, y_pred)
            print("Confusion Matrix \n {} \n".format(temp_confusion_matrix))
            print("Classification Report \n {} \n".format(temp_classification_report))
            disp = plot_confusion_matrix(self.Classifier, self.test_data, self.test_labels, values_format=".4g")
            disp.figure_.suptitle("Confusion Matrix")
            print("TESTING")
            print(disp.confusion_matrix)
            plt.savefig("DecisionTree_Results/Decision_Tree_Images/Base_DecisionTree"
                        "_{}_ConfusionMatrix_{}.png".format(self.DataSetName, datetime.now().strftime("%m-%d~%I%M %p")))
            
            if self.useGridSearch:
                self.RunGridSearch()
            print("Testing of the classifier has finished\n")
        except Exception as TestClassifierException:
            print("Exception occurred while testing the classifier. \n", TestClassifierException)
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
    
    def RunGridSearch(self):
        try:
            
            if self.train_data.shape[0] > 5000 and self.ClassifierName.lower() != "decision tree":
                print("Sorry dataset is too large to run GridSearch and/or Cross-Validation.\n")
                return
            print("Starting Grid Search")
            tree_clf = GridSearchCV(DecisionTreeClassifier(), param_grid=self.GridSearchParameters,
                                    scoring='accuracy', verbose=self.verbose_int, cv=self.cv_folds)
            with parallel_backend('threading'):
                tree_clf.fit(self.train_data, self.train_labels)
            cross_validation_results = pd.DataFrame(tree_clf.cv_results_)
            self.Result_CSV_path = "DecisionTree_Results/DecisionTree" \
                                   "_{}_GridSearch_Results_{}.csv"\
                .format(self.DataSetName, datetime.now().strftime("%m-%d~%I%M %p"))
            cross_validation_results.to_csv(self.Result_CSV_path)
            print("Grid Search has ended.\n")
        
        except Exception as RunGridSearchException:
            print("Exception occurred while running GridSearch. \n", RunGridSearchException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    
    def prune_index(self, index, threshold):
        try:
            """
            prune_index(dt.tree_, 0, 3)
            https://stackoverflow.com/questions/49428469/pruning-decision-trees
            """
            
            if self.Classifier.tree_.value[index][0].sum() < threshold:
                # turn node into a leaf by "unlinking" its children
                self.Classifier.tree_.children_left[index] = TREE_LEAF
                self.Classifier.tree_.children_right[index] = TREE_LEAF
            # if there are children, visit them as well
            if self.Classifier.tree_.children_left[index] != TREE_LEAF:
                self.prune_index(self.Classifier.tree_.children_left[index], threshold)
                self.prune_index(self.Classifier.tree_.children_right[index], threshold)
            
        except Exception as err:
            print("Exception occurred while pruning Decision Tree. \n", err)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
