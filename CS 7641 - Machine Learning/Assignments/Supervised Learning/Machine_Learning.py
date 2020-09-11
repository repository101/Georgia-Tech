import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.svm as svm
import tensorflow as tf
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras


# noinspection PyMethodMayBeStatic,PyPep8Naming,PyUnresolvedReferences
class SupervisedLearning:
    def __init__(self, data=None, x=None, y=None, parameters=None):
        plt.style.use('ggplot')
        self.data = data
        self.parameters = parameters
        self.random_number = 42
        self.overwrite = False
        self.TESTING = True
        self.num_columns = 0
        self.result_template = {"Training Set Accuracy": "", "Testing Set Accuracy": "", "Precision": "",
                                "Recall": "", "F1-Score": "", "Training Time": "", "Evaluation Time": "",
                                "Cross Validation Min": np.nan, "Cross Validation Max": np.nan}
        self.X = x.to_numpy(dtype=np.uint8)
        self.y = y.to_numpy(dtype=np.uint8)
        self.training_set = {"X": None, "y": None}
        self.testing_set = {"X": None, "y": None}
        self.validation_set = {"X": None, "y": None}
        self.static_settings = {"verbose": 10}
        self.Normalize = False
        self.mnist_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
        self.fashion_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                              "Sneaker", "Bag", "Ankle boot"]
        self.decision_tree_dataframe = None
        self.neural_network_dataframe = None
        self.boosting_dataframe = None
        self.svm_dataframe = None
        self.knn_dataframe = None
        self.train_test_split()
        self.samples = self.training_set["X"].shape[0]
        self.number_splits = 3
        self.orchestrator(num_splits=self.number_splits)
    
    def initialize_result_dataframes(self, splits=10):
        idx = [int(np.float(i * self.samples)) for i in np.linspace(0.1, 1.0, splits)]
        cols = ["Training Set Accuracy", "Testing Set Accuracy",
                "Precision", "Recall", "F1-Score", "Training Time",
                "Evaluation Time", "Cross Validation Min", "Cross Validation Max"]
        self.num_columns = len(cols)
        generic_df = pd.DataFrame(data=np.zeros(shape=(len(idx), self.num_columns)), columns=cols, index=idx)
        self.decision_tree_dataframe = generic_df.copy()
        self.neural_network_dataframe = generic_df.copy()
        self.boosting_dataframe = generic_df.copy()
        self.svm_dataframe = generic_df.copy()
        self.knn_dataframe = generic_df.copy()
        return idx
    
    def train_test_split(self):
        
        
        self.training_set["X"], self.testing_set["X"], \
        self.training_set["y"], self.testing_set["y"] = train_test_split(self.X, self.y,
                                                                         test_size=0.20,
                                                                         random_state=self.random_number)
        
        if self.Normalize:
            self.training_set["X"] = self.training_set["X"] / 255.0
            self.training_set["X"] = self.training_set["X"].astype(np.float32)
            self.testing_set["X"] = self.testing_set["X"] / 255.0
            self.testing_set["X"] = self.testing_set["X"].astype(np.float32)
        self.testing_set["X"], self.validation_set["X"] = np.split(self.testing_set["X"].astype(np.float32), 2)
        self.testing_set["y"], self.validation_set["y"] = np.split(self.testing_set["y"].astype(np.uint8), 2)
        return
    
    def save_result_dataframe(self, frame_name, save_name, save_directory="results_dataframe"):
        frame_name = frame_name.lower()
        dataset_path = f"{os.getcwd()}/{save_directory}"
        result_dataframe = None
        try:
            if frame_name in ["nn", "dt", "svm", "boost", "knn", "test"]:
                if frame_name == "nn":
                    result_dataframe = self.neural_network_dataframe
                elif frame_name == "dt":
                    result_dataframe = self.decision_tree_dataframe
                elif frame_name == "svm":
                    result_dataframe = self.svm_dataframe
                elif frame_name == "boost":
                    result_dataframe = self.boosting_dataframe
                elif frame_name == "knn":
                    result_dataframe = self.decision_tree_dataframe
                elif frame_name == "test":
                    result_dataframe = self.data
            else:
                print("Dataframe not found")
            
            if result_dataframe.empty:
                print("Unable to save dataframe. \n \t Reason: Dataframe is empty.")
                return
            else:
                print(f"Saving {frame_name} dataframe to:\n\t'../{save_directory}/{save_name}.feather'")
                result_dataframe.to_feather(f"{dataset_path}/{save_name}_results.feather")
                time.sleep(0.1)
                print(f"\tFinished saving {frame_name} dataframe")
                return
        except Exception as SaveException:
            print("Exception when attempting to access result dataframe.\n", SaveException)
    
    def split_arrays_by_percent(self, percentage):
        if percentage is 1.0 or percentage is 1:
            return self.training_set["X"], self.training_set["y"], \
                   self.testing_set["X"], self.testing_set["y"], \
                   self.validation_set["X"], self.validation_set["y"]
        
        # find proper index for percentage of arrays.
        try:
            train_limit = int(np.floor(self.training_set["X"].shape[0] * percentage))
            test_limit = int(np.floor(self.testing_set["X"].shape[0] * percentage))
            validation_limit = int(np.floor(self.validation_set["X"].shape[0] * percentage))
            
            Xtrain = self.training_set["X"][:train_limit, :]
            Ytrain = self.training_set["y"][:train_limit]
            
            Xtest = self.testing_set["X"][:test_limit, :]
            Ytest = self.testing_set["y"][:test_limit]
            
            Xvalid = self.validation_set["X"][:validation_limit, :]
            Yvalid = self.validation_set["y"][:validation_limit]
            
            return Xtrain, Ytrain, Xtest, Ytest, Xvalid, Yvalid
        except Exception as Split_Array_Exception:
            print("Exception while attempting to split arrays by percentage. \n", Split_Array_Exception)
    
    def orchestrator(self, num_splits=10):
        df_index = self.initialize_result_dataframes(splits=num_splits)
        percents = np.linspace(0.1, 1.0, num_splits)
        for i in range(len(df_index)):
            nn_results = self.get_neural_network_results(percent=percents[i])
            self.neural_network_dataframe.loc[df_index[i]] = nn_results.values()
            print()
        # dt_results = self.get_decision_tree_results(percent=percents[i])
        # self.decision_tree_dataframe.loc[df_index[i]] = dt_results.values()
        knn_results = self.get_knn_results(percent=0.5)
        self.knn_dataframe.loc[df_index[i]] = knn_results.values()
        svm_results = self.get_svm_results(percent=1.0)
        self.svm_dataframe.loc[df_index[i]] = svm_results.values()
        boost_results = self.get_boosting_results(percent=1.0)
        self.boosting_dataframe.loc[df_index[i]] = boost_results.values()
        # self.generate_image_grid(class_names=self.mnist_names)
        # self.neural_network_dataframe.reset_index(inplace=True)
        # self.neural_network_dataframe.to_feather(f"{os.getcwd()}/test_df.feather")
        
        return
    
    def generate_image_grid(self, class_names):
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.training_set["X"][i].reshape((28, 28)), cmap=plt.cm.binary)
            plt.xlabel(class_names[self.training_set["y"][i]])
        plt.savefig("Image_Grid.png")
        return
    
    def evaluate_training_times(self, clf, Xtrain, Ytrain):
        start_time = time.time()
        clf.fit(Xtrain, Ytrain)
        end_time = time.time()
        training_time = end_time - start_time
        
        return training_time, clf
    
    def evaluate_classifiers(self, clf, Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation):
        results = np.zeros(shape=(self.num_columns,))
        
        training_time, clf = self.evaluate_training_times(clf=clf, Xtrain=Xtrain, Ytrain=Ytrain)
        
        evaluation_time, Ypred = self.get_evaluation_times(clf=clf, X_array=Xtest)
        accuracy = metrics.accuracy_score(Ytest, Ypred)
        print("Accuracy:", )
    
    def get_evaluation_times(self, clf, X_array):
        start_time = time.time()
        ypred = clf.predict(X_array)
        end_time = time.time()
        evaluation_time = end_time - start_time
        return evaluation_time, ypred
    
    def get_decision_tree_results(self, percent=1.0):
        """
        Decision Trees. For the decision tree, you should implement or steal a decision tree algorithm
           (and by "implement or steal" I mean "steal"). Be sure to use some form of pruning. You are not required
           to use information gain (for example, there is something called the GINI index that is sometimes used)
           to split attributes, but you should describe whatever it is that you do use.

        :param percent: a value to control size of arrays for learner
        :type percent: float
        :return:
        :rtype:
        """
        results = self.result_template.copy()
        
        Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation = self.split_arrays_by_percent(percentage=percent)
        
        dt = DecisionTreeClassifier(max_depth=self.parameters["DT"]["max_depth"],
                                    min_samples_leaf=self.parameters["DT"]["min_samples_leaf"],
                                    min_samples_split=self.parameters["DT"]["min_samples_split"],
                                    random_state=self.random_number)
        # region Training Time
        start_time = time.time()
        dt.fit(Xtrain, Ytrain)
        end_time = time.time()
        elapsed_time = end_time - start_time
        results["Training Time"] = elapsed_time
        # endregion
        
        # region Evaluation Time
        start_time = time.time()
        y_pred = dt.predict(Xtest)
        end_time = time.time()
        elapsed_time = end_time - start_time
        results["Evaluation Time"] = elapsed_time
        # endregion
        
        # region Precision
        results["Precision"] = precision_score(y_true=Ytest, y_pred=y_pred, average='macro')
        # endregion
        
        # region Recall
        results["Recall"] = recall_score(y_true=Ytest, y_pred=y_pred, average='macro')
        # endregion
        
        # region F1-Score
        results["F1-Score"] = f1_score(y_true=Ytest, y_pred=y_pred, average='macro')
        # endregion
        
        # region Testing Accuracy
        results["Testing Set Accuracy"] = accuracy_score(y_true=Ytest, y_pred=y_pred)
        # endregion
        
        # region Training Accuracy
        train_pred = dt.predict(Xtrain)
        results["Training Set Accuracy"] = accuracy_score(y_true=Ytrain, y_pred=train_pred)
        # endregion
        
        # self.plot_learning_curve(clf, "Decision Tree", X=Xtrain, y=Ytrain, ylim=(0.7, 1.01), cv=5, save_individual=True)
        
        # self.save_result_dataframe(frame_name="dt", save_name="DT")
        return results
    
    def get_neural_network_results(self, percent=1.0):
        """
        Neural Networks. For the neural network you should implement or steal your favorite kind of network
           and training algorithm. You may use networks of nodes with as many layers as you like and
           any activation function you see fit.

        :param percent: a value to control size of arrays for learner
        :type percent: float
        :return:
        :rtype:
        """
        results = self.result_template.copy()
        
        Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation = self.split_arrays_by_percent(percentage=percent)
        
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(784,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10)
        ])
        
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        
        # region Training Time
        start_time = time.time()
        model_results = model.fit(Xtrain, Ytrain, epochs=self.parameters["NN"]["epochs"], verbose=2,
                                  validation_data=(self.validation_set["X"], self.validation_set["y"]))
        cv = np.asarray(model_results.history["accuracy"])
        results["Cross Validation Min"] = cv.min()
        results["Cross Validation Max"] = cv.max()
        end_time = time.time()
        elapsed_time = end_time - start_time
        results["Training Time"] = elapsed_time
        # endregion
        
        # region Evaluation Time
        start_time = time.time()
        y_pred = np.argmax(model.predict(Xtest), axis=-1)
        end_time = time.time()
        elapsed_time = end_time - start_time
        results["Evaluation Time"] = elapsed_time
        # endregion
        
        # region Precision
        results["Precision"] = precision_score(y_true=Ytest, y_pred=y_pred, average='macro')
        # endregion
        
        # region Recall
        results["Recall"] = recall_score(y_true=Ytest, y_pred=y_pred, average='macro')
        # endregion
        
        # region F1-Score
        results["F1-Score"] = f1_score(y_true=Ytest, y_pred=y_pred, average='macro')
        # endregion
        
        # region Testing Accuracy
        results["Testing Set Accuracy"] = accuracy_score(y_true=Ytest, y_pred=y_pred)
        # endregion
        
        # region Training Accuracy
        results["Training Set Accuracy"] = np.asarray(model_results.history["accuracy"]).mean()
        # endregion
        
        # self.save_result_dataframe(frame_name="nn", save_name="NN")
        return results
    
    def get_boosting_results(self, percent=1.0):
        """
        Boosting. Implement or steal a boosted version of your decision trees. As before, you will want to use some
           form of pruning, but presumably because you're using boosting you can afford to be much more aggressive
           about your pruning.

        :param percent: a value to control size of arrays for learner
        :type percent: float
        :return:
        :rtype:
        """
        results = self.result_template.copy()
        Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation = self.split_arrays_by_percent(percentage=percent)
        
        boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                                   n_estimators=self.parameters["Boost"]["n_estimators"],
                                   learning_rate=self.parameters["Boost"]["learning_rate"],
                                   algorithm=self.parameters["Boost"]["algorithm"])
        
        # region Training Time
        start_time = time.time()
        boost.fit(Xtrain, Ytrain)
        end_time = time.time()
        elapsed_time = end_time - start_time
        results["Training Time"] = elapsed_time
        # endregion
        
        # region Evaluation Time
        start_time = time.time()
        y_pred = boost.predict(Xtest)
        end_time = time.time()
        elapsed_time = end_time - start_time
        results["Evaluation Time"] = elapsed_time
        # endregion
        
        # region Precision
        results["Precision"] = precision_score(y_true=Ytest, y_pred=y_pred, average='macro')
        # endregion
        
        # region Recall
        results["Recall"] = recall_score(y_true=Ytest, y_pred=y_pred, average='macro')
        # endregion
        
        # region F1-Score
        results["F1-Score"] = f1_score(y_true=Ytest, y_pred=y_pred, average='macro')
        # endregion
        
        # region Testing Accuracy
        results["Testing Set Accuracy"] = accuracy_score(y_true=Ytest, y_pred=y_pred)
        # endregion
        
        # region Training Accuracy
        train_pred = boost.predict(Xtrain)
        results["Training Set Accuracy"] = accuracy_score(y_true=Ytrain, y_pred=train_pred)
        # endregion
        
        # self.save_result_dataframe(frame_name="boost", save_name="Boosting")
        return results
    
    def get_svm_results(self, percent=1.0):
        """
        Support Vector Machines. You should implement (for sufficiently loose definitions of implement including
           "download") SVMs. This should be done in such a way that you can swap out kernel functions.
           I'd like to see at least two.

        C default=1.0 Regularization parameter. The strength of the
                      regularization is inversely proportional to C. Must be strictly positive.
                      The penalty is a squared l2 penalty.

        kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’ Specifies the
              kernel type to be used in the algorithm. It must be one of ‘linear’,
              ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given,
              ‘rbf’ will be used. If a callable is given it is used to pre-compute
              the kernel matrix from data matrices; that matrix should be an array
              of shape (n_samples, n_samples).

        degree default=3 Degree of the polynomial kernel function (‘poly’).
              Ignored by all other kernels.

        decision_function_shape{‘ovo’, ‘ovr’}, default=’ovr’
             Whether to return a one-vs-rest (‘ovr’) decision function of shape
             (n_samples, n_classes) as all other classifiers, or the original
             one-vs-one (‘ovo’) decision function of libsvm which has shape
             (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one (‘ovo’)
             is always used as multi-class strategy. The parameter is ignored for binary
             classification.

        :param percent: a value to control size of arrays for learner
        :type percent: float
        :return:
        :rtype:
        """
        results = self.result_template.copy()
        
        Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation = self.split_arrays_by_percent(percentage=percent)
        
        svm = svm.SVC(C=self.parameters["SVM"]["C"],
                      kernel=self.parameters["SVM"]["kernel"],
                      degree=self.parameters["SVM"]["degree"],
                      decision_function_shape=self.parameters["SVM"]["decision_function_shape"])
        
        # region Training Time
        start_time = time.time()
        svm.fit(Xtrain, Ytrain)
        end_time = time.time()
        elapsed_time = end_time - start_time
        results["Training Time"] = elapsed_time
        # endregion
        
        # region Evaluation Time
        start_time = time.time()
        y_pred = svm.predict(Xtest)
        end_time = time.time()
        elapsed_time = end_time - start_time
        results["Evaluation Time"] = elapsed_time
        # endregion
        
        # region Precision
        results["Precision"] = precision_score(y_true=Ytest, y_pred=y_pred, average='macro')
        # endregion
        
        # region Recall
        results["Recall"] = recall_score(y_true=Ytest, y_pred=y_pred, average='macro')
        # endregion
        
        # region F1-Score
        results["F1-Score"] = f1_score(y_true=Ytest, y_pred=y_pred, average='macro')
        # endregion
        
        # region Testing Accuracy
        results["Testing Set Accuracy"] = accuracy_score(y_true=Ytest, y_pred=y_pred)
        # endregion
        
        # region Training Accuracy
        train_pred = svm.predict(Xtrain)
        results["Training Set Accuracy"] = accuracy_score(y_true=Ytrain, y_pred=train_pred)
        # endregion
        
        return results
    
    def get_knn_results(self, percent=1.0):
        """
        k-Nearest Neighbors. You should "implement" (the quotes mean I don't mean it: steal the code)
           kNN. Use different values of k.

        n_neighbors int, default=5
        algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’

        :param percent: a value to control size of arrays for learner
        :type percent: float
        :return:
        :rtype:
        """
        results = self.result_template.copy()
        
        Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation = self.split_arrays_by_percent(percentage=percent)
        
        knn = KNeighborsClassifier(n_neighbors=self.parameters["KNN"]["n_neighbors"],
                                   algorithm=self.parameters["KNN"]["algorithm"],
                                   leaf_size=self.parameters["KNN"]["leaf_size"])
        knn.fit(Xtrain, Ytrain)
        
        plt, train_sizes, train_scores, \
        test_scores, fit_times = self.plot_learning_curve(estimator=knn, title="K-Nearest Neighbor",
                                                          X=Xtrain, y=Ytrain,
                                                          train_sizes=np.linspace(.1, 1.0, self.number_splits))
        
        # # region Training Time
        # start_time = time.time()
        # knn.fit(Xtrain, Ytrain)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # results["Training Time"] = elapsed_time
        # # endregion
        #
        # # region Evaluation Time
        # start_time = time.time()
        # y_pred = knn.predict(Xtest)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # results["Evaluation Time"] = elapsed_time
        # # endregion
        #
        # # region Precision
        # results["Precision"] = precision_score(y_true=Ytest, y_pred=y_pred, average='macro')
        # # endregion
        #
        # # region Recall
        # results["Recall"] = recall_score(y_true=Ytest, y_pred=y_pred, average='macro')
        # # endregion
        #
        # # region F1-Score
        # results["F1-Score"] = f1_score(y_true=Ytest, y_pred=y_pred, average='macro')
        # # endregion
        #
        # # region Testing Accuracy
        # results["Testing Set Accuracy"] = accuracy_score(y_true=Ytest, y_pred=y_pred)
        # # endregion
        #
        # # region Training Accuracy
        # train_pred = knn.predict(Xtrain)
        # results["Training Set Accuracy"] = accuracy_score(y_true=Ytrain, y_pred=train_pred)
        # # endregion
        
        return results
    
    def plot_learning_curve(self, estimator, title, X, y, axes=None, ylim=None, cv=None,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), save_individual=False):
        """
        FROM https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

        Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        axes : array of 3 axes, optional (default=None)
            Axes to use for plotting the curves.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

              - None, to use the default 5-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, optional (default=None)
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        train_sizes : array-like, shape (n_ticks,), dtype float or int
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the dtype is float, it is regarded as a
            fraction of the maximum size of the training set (that is determined
            by the selected validation method), i.e. it has to be within (0, 1].
            Otherwise it is interpreted as absolute sizes of the training sets.
            Note that for classification the number of samples usually have to
            be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        if self.TESTING:
            verbose = True
            verbose_val = 10
        else:
            verbose = False
            verbose_val = 0
        
        with joblib.parallel_backend("threading", n_jobs=-1):
            train_sizes, train_scores, test_scores, fit_times, _ = \
                learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
                               return_times=True, verbose=verbose_val)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
        
        if save_individual:
            plt.close("all")
            plt.grid()
            plt.grid(which='major', linestyle='-', linewidth='0.5', color='white')
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
            plt.legend(loc="best")
            plt.savefig(f"{os.getcwd()}/figures/{title}_Learning_Curve.png")
            
            plt.close("all")
            plt.grid()
            plt.grid(which='major', linestyle='-', linewidth='0.5', color='white')
            plt.plot(train_sizes, fit_times_mean, 'o-')
            plt.fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
            plt.xlabel("Training examples", fontsize=15, weight='bold')
            plt.ylabel("fit_times", fontsize=15, weight='bold')
            plt.title("Scalability of the model", fontsize=15, weight='bold')
            plt.savefig(f"{os.getcwd()}/figures/{title}_Fit_Times.png")
            
            plt.close("all")
            plt.grid()
            plt.grid(which='major', linestyle='-', linewidth='0.5', color='white')
            plt.plot(fit_times_mean, test_scores_mean, 'o-')
            plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
            plt.xlabel("fit_times", fontsize=15, weight='bold')
            plt.ylabel("Score", fontsize=15, weight='bold')
            plt.title("Performance of the model", fontsize=15, weight='bold')
            plt.savefig(f"{os.getcwd()}/figures/{title}_Fit_Times_Vs_Score.png")
            return plt, train_sizes, train_scores, test_scores, fit_times
        
        else:
            if axes is None:
                _, axes = plt.subplots(1, 3, figsize=(20, 5))
            
            axes[0].set_title(title, fontsize=15, weight='bold')
            if ylim is not None:
                axes[0].set_ylim(*ylim)
            axes[0].set_xlabel("Training examples", fontsize=15, weight='bold')
            axes[0].set_ylabel("Score", fontsize=15, weight='bold')
            
            # Plot learning curve
            axes[0].grid()
            # Customize the major grid
            axes[0].grid(which='major', linestyle='-', linewidth='0.5', color='white')
            
            axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.1,
                                 color="r")
            axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1,
                                 color="g")
            axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                         label="Training score")
            axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                         label="Cross-validation score")
            axes[0].legend(loc="best")
            
            # Plot n_samples vs fit_times
            axes[1].grid()
            axes[1].plot(train_sizes, fit_times_mean, 'o-')
            axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                                 fit_times_mean + fit_times_std, alpha=0.1)
            axes[1].set_xlabel("Training examples", fontsize=15, weight='bold')
            axes[1].set_ylabel("fit_times", fontsize=15, weight='bold')
            axes[1].set_title("Scalability of the model", fontsize=15, weight='bold')
            
            # Plot fit_time vs score
            axes[2].grid()
            axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
            axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1)
            axes[2].set_xlabel("fit_times", fontsize=15, weight='bold')
            axes[2].set_ylabel("Score", fontsize=15, weight='bold')
            axes[2].set_title("Performance of the model", fontsize=15, weight='bold')
            
            plt.savefig(f"{os.getcwd()}/figures/{title}.png")
            return plt, train_sizes, train_scores, test_scores, fit_times

        
    
