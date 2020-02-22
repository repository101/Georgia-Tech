import os
import pathlib
import sys
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import parallel_backend

Dataset_Splits = 1


def DecisionTreeTrainingSizeCompare(training_data, training_labels, testing_data,
                                    testing_labels, dataset):
    try:
        if training_data is None \
                or training_labels is None \
                or testing_data is None \
                or testing_labels is None:
            return
        """
        index 0 = 25% of training set
        index 1 = 50% of training set
        index 2 = 75% of training set
        index 3 = 100% of training set
        """

        directory = "DecisionTree_Results/Decision_Tree_Images/"
        clf_type = 'DecisionTree'
        cwd = pathlib.Path().absolute()
        number_of_data_entries = training_data.shape[0]
        run_time = [0.0]
        accuracy = [0.0]
        sizes = [0.0]
        start = 1
        stop = Dataset_Splits
        percent = (start / stop)
        for i in range(start, stop + 1, 1):
            plt.close('all')
            size = i * percent
            sizes.append(int(size * 100))
            Classifier = DecisionTreeClassifier(criterion="entropy", max_depth=50,
                                                min_samples_leaf=6, min_samples_split=2)
            set_0 = {"Training_Data": training_data[:int(number_of_data_entries * size)],
                     "Training_Labels": training_labels[:int(number_of_data_entries * size)]}
            start_time = timer()
            with parallel_backend('threading'):
                Classifier.fit(set_0["Training_Data"], set_0["Training_Labels"])
            end_time = timer()
            elapsed_time = end_time - start_time
            run_time.append(elapsed_time)
            y_pred = Classifier.predict(testing_data)
            temp_confusion_matrix = confusion_matrix(testing_labels, y_pred)
            temp_classification_report = classification_report(testing_labels, y_pred)
            size_str = str(int(size * 100))
            print("Confusion Matrix \n {} \n".format(temp_confusion_matrix))
            print("Classification Report \n {} \n".format(temp_classification_report))
            classifier_score = Classifier.score(testing_data, testing_labels)
            accuracy.append(classifier_score)
            with plt.style.context('bmh'):
                disp = plot_confusion_matrix(Classifier, testing_data, testing_labels, values_format=".4g")
                disp.figure_.suptitle("Confusion Matrix {}% Training Data".format(size_str))
                print("TESTING {}%".format(size_str))
                print(disp.confusion_matrix)
                dir = "{}/{}/{}_" \
                      "{}_ConfusionMatrix_{}%_Training.png".format(cwd, directory, clf_type, dataset, size_str)
                plt.savefig(dir)
            print(classifier_score)

        with plt.style.context('bmh'):
            fig, ax = plt.subplots()
            ax.set_xlabel("Percent of Training Set")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Accuracy vs Training Set Size")
    
            ax.plot([i for i in range(0,
                                      int((stop * percent) * 100) + 1,
                                      int((start / stop) * 100))],
                    accuracy, marker='o', label="accuracy")
            ax.legend()
            dir = "{}/{}/{}_" \
                  "{}_Training_Set_Size_Impact.png".format(cwd, directory, clf_type, dataset)
            plt.savefig(dir)

        with plt.style.context('ggplot'):
            fig0, ax0 = plt.subplots()
            ax0.set_xlabel("Percent of Training Set")
            ax0.set_ylabel("Accuracy (%)", color='tab:green')
            ax0.set_title("Accuracy vs Training Set Size")
            ax0.plot([i for i in range(0, int((stop * percent) * 100) + 1, int((start / stop) * 100))],
                     accuracy, "tab:green", marker='o', label="accuracy")
            ax0.tick_params(axis='y', labelcolor="tab:green")
            
            ax3 = ax0.twinx()
            ax3.set_ylabel("Training Time (s)", color="tab:blue")
            ax3.plot([i for i in range(0, int((stop * percent) * 100) + 1, int((start / stop) * 100))],
                     run_time, "tab:blue", marker='o', label="training-time")
            ax3.tick_params(axis='y', labelcolor="tab:blue")
            fig0.tight_layout()
            dir = "{}/{}/{}_{}" \
                  "_Training_Set_Size_Impact_vs_Training_Time.png".format(cwd, directory, clf_type, dataset)
            plt.savefig(dir)
            
    except Exception as err:
        print("Exception occurred while comparing how training size impacts accuracy. \n", err)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def DecisionTreeDepthCompare(training_data, training_labels, testing_data, testing_labels, dataset, tree_max_depth=50):
    try:
        if training_data is None \
                or training_labels is None \
                or testing_data is None \
                or testing_labels is None:
            return
        """
        https://towardsdatascience.com/understanding-decision-trees-for-classification-python-9663d683c952
        by Michael Galarnyk
        """
        accuracy = [0.0]
        cwd = pathlib.Path().absolute()
        run_time = [0.0]
        
        for depth in range(1, tree_max_depth + 1):
            clf = DecisionTreeClassifier(max_depth=depth, criterion='entropy')
            start_time = timer()
            with parallel_backend('threading'):
                clf.fit(training_data, training_labels)
            end_time = timer()
            elapsed_time = end_time - start_time
            run_time.append(elapsed_time)
            score = clf.score(testing_data, testing_labels)
            print("Current Depth Tree {} / {} ".format(depth, tree_max_depth))
            print("Accuracy: {0:.3f}%\n".format(score))
            accuracy.append(score)
        results = np.asarray(accuracy)
        results.tofile("{}/DecisionTree_Results/DecisionTree_"
                       "{}_Depth_Compare.csv".format(cwd, dataset), sep=",", format="%.4f")
        with plt.style.context('ggplot'):
            fig, ax = plt.subplots()
            ax.set_xlabel("Tree Depth")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Accuracy vs Tree Depth for Testing sets")
            ax.plot([i for i in range(tree_max_depth + 1)], accuracy, marker='o', label="test")
            ax.legend()
            plt.savefig("{}/DecisionTree_Results/Decision_Tree_Images/DecisionTree"
                        "_{}_Tree_Depth_VS_Accuracy.png".format(cwd, dataset))
            
        with plt.style.context('ggplot'):
            fig1, ax1 = plt.subplots()
            ax1.set_xlabel("Tree Depth")
            ax1.set_ylabel("Accuracy (%)", color='tab:red')
            ax1.plot([i for i in range(tree_max_depth + 1)], accuracy, "tab:red", marker='o', label="accuracy")
            ax1.set_title("Accuracy vs Tree Depth vs Training Time")
            ax1.tick_params(axis="y", labelcolor="tab:red")

            ax2 = ax1.twinx()
            ax2.set_ylabel("Training Time (s)", color="tab:blue")
            ax2.plot([i for i in range(tree_max_depth + 1)], run_time, "tab:blue", marker='o', label="training-time")
            ax2.tick_params(axis="y", labelcolor="tab:blue")
            fig1.tight_layout()
            plt.savefig("{}/DecisionTree_Results/Decision_Tree_Images/DecisionTree_"
                        "{}_Tree_Depth_VS_Accuracy_VS"
                        "_Training_Time.png".format(cwd, dataset))

        print()
    except Exception as err:
        print("Exception occurred while comparing Decision Tree depths impact on accuracy. \n", err)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def SVMTrainingSizeCompare(training_data, training_labels, testing_data, testing_labels, dataset):
    try:
        if training_data is None \
                or training_labels is None \
                or testing_data is None \
                or testing_labels is None:
            return
        """
        index 0 = 25% of training set
        index 1 = 50% of training set
        index 2 = 75% of training set
        index 3 = 100% of training set
        """
        directory = "SVM_Results/Support_Vector_Machine_Images/"
        clf_type = "SupportVectorMachine"
        cwd = pathlib.Path().absolute()
        Scaler = StandardScaler().fit(training_data)
        
        training_data = Scaler.transform(training_data)
        testing_data = Scaler.transform(testing_data)

        number_of_data_entries = training_data.shape[0]
        run_time = [[0.0], [0.0], [0.0], [0.0]]
        kernels = ["Linear", "RBF", "Poly", "Sigmoid"]
        colors = ["tab:blue", "tab:green", "tab:red", "tab:orange"]
        accuracy = [[0.0], [0.0], [0.0], [0.0]]
        sizes = [0.0]
        start = 1
        stop = Dataset_Splits
        percent = (start / stop)
        for i in range(start, stop + 1, 1):
            plt.close('all')
            size = i * percent
            sizes.append(int(size * 100))
            set_0 = {"Training_Data": training_data[:int(number_of_data_entries * size)],
                     "Training_Labels": training_labels[:int(number_of_data_entries * size)]}
            count = 0
            for kernel in kernels:
                Classifier = svm.SVC(kernel=kernel.lower(), verbose=True, max_iter=7000)
                start_time = timer()
                with parallel_backend('threading'):
                    Classifier.fit(set_0["Training_Data"], set_0["Training_Labels"])
                end_time = timer()
                elapsed_time = end_time - start_time
                run_time[count].append(elapsed_time)
                y_pred = Classifier.predict(testing_data)
                temp_confusion_matrix = confusion_matrix(testing_labels, y_pred)
                temp_classification_report = classification_report(testing_labels, y_pred)
                size_str = str(int(size * 100))
                print("Confusion Matrix \n {} \n".format(temp_confusion_matrix))
                print("Classification Report \n {} \n".format(temp_classification_report))
                classifier_score = Classifier.score(testing_data, testing_labels)
                accuracy[count].append(classifier_score)
                with plt.style.context('bmh'):
                    disp = plot_confusion_matrix(Classifier, testing_data, testing_labels, values_format=".4g")
                    disp.figure_.suptitle("Confusion Matrix {}% Training Data".format(size_str))
                    print("TESTING {}%".format(size_str))
                    print(disp.confusion_matrix)
                    plt.savefig("{}/{}/{}_{}%_{}_ConfusionMatrix.png"
                                .format(cwd, directory, kernel, size_str, dataset))
                print(classifier_score)
                count += 1
        
        for i in range(len(kernels)):
            with plt.style.context('bmh'):
                fig, ax = plt.subplots()
                ax.set_xlabel("Percent of Training Set")
                ax.set_ylabel("Accuracy (%)")
                ax.set_title("Accuracy vs Training Set Size ({})")
                
                ax.plot([i for i in range(0,
                                          int((stop * percent) * 100) + 1,
                                          int((start / stop) * 100))],
                        accuracy[i], marker='o', label=kernels[i])
                ax.legend()
                plt.savefig("{}/{}/{}_{}_Training_Size_Impact.png"
                            .format(cwd, directory, kernels[i], dataset))
        x = [i for i in range(0, int((stop * percent) * 100) + 1, int((start / stop) * 100))]
        with plt.style.context('ggplot'):
            fig0, ax0 = plt.subplots()
            ax0.set_xlabel("Percent of Training Set")
            ax0.set_ylabel("Accuracy (%)", color='tab:green')
            ax0.set_title("Accuracy vs Training Set Size ('linear', 'rbf', 'poly', 'sigmoid')")
            
            for i in range(len(kernels)):
                y1 = accuracy[i]
                ax0.plot(x, y1, colors[i], marker='o', label=kernels[i])

            ax0.tick_params(axis='y', labelcolor="tab:green")
            ax0.legend(title="Accuracy", loc=2)
            ax3 = ax0.twinx()
            ax3.set_ylabel("Training Time (s)", color="tab:blue")
            for i in range(len(kernels)):
                y2 = run_time[i]
                ax3.plot(x, y2, colors[i], label=kernels[i], linestyle="--")
            ax3.legend(title="Training Time", loc=4)
            ax3.tick_params(axis='y', labelcolor="tab:blue")
            fig0.tight_layout()
            plt.savefig("{}/{}/{}_{}_Training_Size_vs_Time.png"
                        .format(cwd, directory, clf_type, dataset))
        print()
    
    except Exception as err:
        print("Exception occurred while comparing how training size impacts accuracy. \n", err)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def NeuralNetworkTrainingSizeCompare(training_data, training_labels, testing_data, testing_labels, dataset):
    try:
        if training_data is None \
                or training_labels is None \
                or testing_data is None \
                or testing_labels is None:
            return

        """
        index 0 = 25% of training set
        index 1 = 50% of training set
        index 2 = 75% of training set
        index 3 = 100% of training set
        """
        directory = "NeuralNetwork_Results/Neural_Network_Images/"
        clf_type = "NeuralNetwork"
        cwd = pathlib.Path().absolute()
        Scaler = StandardScaler().fit(training_data)

        training_data = Scaler.transform(training_data)
        testing_data = Scaler.transform(testing_data)

        number_of_data_entries = training_data.shape[0]
        adam_run_time = [0.0]
        sgd_run_time = [0.0]
        solvers = ["SGD", "Adam"]
        colors = ["green", "red", "blue", "orange"]
        accuracy = [[0.0], [0.0]]
        classifier_list = []
        sizes = [0.0]
        start = 1
        stop = Dataset_Splits
        percent = (start / stop)
        for i in range(start, stop + 1, 1):
            plt.close('all')
            size = i * percent
            sizes.append(int(size * 100))
            set_0 = {"Training_Data": training_data[:int(number_of_data_entries * size)],
                     "Training_Labels": training_labels[:int(number_of_data_entries * size)]}
            count = 0
            for solver in solvers:
                Classifier = MLPClassifier(solver=solver.lower(), max_iter=200, verbose=3, hidden_layer_sizes=(100,))
                start_time = timer()
                with parallel_backend('threading'):
                    Classifier.fit(set_0["Training_Data"], set_0["Training_Labels"])
                end_time = timer()
                elapsed_time = end_time - start_time
                if solver == "SGD":
                    sgd_run_time.append(elapsed_time)
                else:
                    adam_run_time.append(elapsed_time)
                
                y_pred = Classifier.predict(testing_data)
                temp_confusion_matrix = confusion_matrix(testing_labels, y_pred)
                temp_classification_report = classification_report(testing_labels, y_pred)
                size_str = str(int(size * 100))
                print("Confusion Matrix \n {} \n".format(temp_confusion_matrix))
                print("Classification Report \n {} \n".format(temp_classification_report))
                classifier_score = Classifier.score(testing_data, testing_labels)
                accuracy[count].append(classifier_score)
                with plt.style.context('bmh'):
                    disp = plot_confusion_matrix(Classifier, testing_data, testing_labels, values_format=".4g")
                    disp.figure_.suptitle("Confusion Matrix {}% Training Data ({})".format(size_str, solver))
                    print("TESTING {}%".format(size_str))
                    print(disp.confusion_matrix)
                    plt.savefig("{}/{}/{}_{}%_{}_ConfusionMatrix.png"
                                .format(cwd, directory, solver, size_str, dataset))
                plt.close('all')

                with plt.style.context('bmh'):
                    fig, ax = plt.subplots()
                    ax.set_xlabel("Iterations")
                    ax.set_ylabel("Loss (%)")
                    ax.set_title("Loss vs Iterations ({})".format(solver))
    
                    ax.plot([i for i in range(len(Classifier.loss_curve_))], Classifier.loss_curve_, marker='o',
                            label=solver)
                    ax.legend()
                    plt.savefig("{}/{}/{}_{}%_{}_Training_Size_Impact.png"
                                .format(cwd, directory, solver, size_str, dataset))
                    
                print(classifier_score)
                count += 1

        for i in range(len(solvers)):
            with plt.style.context('bmh'):
                fig, ax = plt.subplots()
                ax.set_xlabel("Percent of Training Set")
                ax.set_ylabel("Accuracy (%)")
                ax.set_title("Accuracy vs Training Set Size ({})".format(solvers[i]))

                ax.plot([i for i in range(0,
                                          int((stop * percent) * 100) + 1,
                                          int((start / stop) * 100))],
                        accuracy[i], marker='o', label=solvers[i])
                ax.legend()
                plt.savefig("{}/{}/{}_{}_Training_Size_Impact.png"
                            .format(cwd, directory, solvers[i], dataset))
                
        for i in range(len(solvers)):
            with plt.style.context('bmh'):
                fig, ax = plt.subplots()
                ax.set_xlabel("Percent of Training Set")
                ax.set_ylabel("Accuracy (%)")
                ax.set_title("Accuracy vs Training Set Size ({})".format(solvers[i]))

                ax.plot([i for i in range(0,
                                          int((stop * percent) * 100) + 1,
                                          int((start / stop) * 100))],
                        accuracy[i], marker='o', label=solvers[i])
                ax.legend()
                plt.savefig("{}/{}/{}_{}_Training_Size_Impact.png"
                            .format(cwd, directory, solvers[i], dataset))
        
        accuracy_array = np.asarray(accuracy)
        adam_run_time_array = np.asarray(adam_run_time)
        sgd_run_time_array = np.asarray(sgd_run_time)
        for i in range(len(accuracy_array)):
            accuracy_array.tofile("{}/{}/Neural_Network_{}_Accuracy.csv"
                                  .format(cwd, directory, solvers[i]), sep=",", format="%.3f")
        adam_run_time_array.tofile("{}/{}/Neural_Network_{}_RunTime.csv"
                                   .format(cwd, directory, 'Adam'), sep=",", format="%.3f")
        sgd_run_time_array.tofile("{}/{}/Neural_Network_{}_RunTime.csv"
                                  .format(cwd, directory, 'SGD'), sep=",", format="%.3f")

    except Exception as err:
        print("Exception occurred while comparing how training size impacts accuracy. \n", err)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def KNNTrainingSizeCompare(training_data, training_labels, testing_data, testing_labels, dataset):
    try:
        if training_data is None \
                or training_labels is None \
                or testing_data is None \
                or testing_labels is None:
            return
        
        """
        index 0 = 25% of training set
        index 1 = 50% of training set
        index 2 = 75% of training set
        index 3 = 100% of training set
        """
        directory = "KNN_Results/K_Nearest_Neighbor_Images/"
        clf_type = "KNearestNeighbor"
        cwd = pathlib.Path().absolute()
        Scaler = StandardScaler().fit(training_data)
        
        training_data = Scaler.transform(training_data)
        testing_data = Scaler.transform(testing_data)
        
        number_of_data_entries = training_data.shape[0]
        ball_tree_run_time = [0.0]
        kd_tree_run_time = [0.0]
        brute_run_time = [0.0]
        algorithms = ['ball_tree', 'kd_tree', 'brute']
        colors = ["green", "red", "blue", "orange"]
        accuracy = [[0.0], [0.0], [0.0]]
        classifier_list = []
        sizes = [0.0]
        start = 1
        stop = Dataset_Splits
        percent = (start / stop)
        for i in range(start, stop + 1, 1):
            plt.close('all')
            size = i * percent
            sizes.append(int(size * 100))
            set_0 = {"Training_Data": training_data[:int(number_of_data_entries * size)],
                     "Training_Labels": training_labels[:int(number_of_data_entries * size)]}
            count = 0
            for algorithm in algorithms:
                Classifier = KNeighborsClassifier()
                start_time = timer()
                with parallel_backend('threading'):
                    Classifier.fit(set_0["Training_Data"], set_0["Training_Labels"])
                end_time = timer()
                elapsed_time = end_time - start_time
                if algorithm == "ball_tree":
                    kd_tree_run_time.append(elapsed_time)
                elif algorithm == "kd_tree":
                    ball_tree_run_time.append(elapsed_time)
                elif algorithm == "brute":
                    brute_run_time.append(elapsed_time)
                else:
                    continue
                    
                y_pred = Classifier.predict(testing_data)
                temp_confusion_matrix = confusion_matrix(testing_labels, y_pred)
                temp_classification_report = classification_report(testing_labels, y_pred)
                size_str = str(int(size * 100))
                print("Confusion Matrix \n {} \n".format(temp_confusion_matrix))
                print("Classification Report \n {} \n".format(temp_classification_report))
                classifier_score = Classifier.score(testing_data, testing_labels)
                accuracy[count].append(classifier_score)
                with plt.style.context('bmh'):
                    disp = plot_confusion_matrix(Classifier, testing_data, testing_labels, values_format=".4g")
                    disp.figure_.suptitle("Confusion Matrix {}% Training Data ({})".format(size_str, algorithm))
                    print("TESTING {}%".format(size_str))
                    print(disp.confusion_matrix)
                    plt.savefig("{}/{}/{}_{}%_{}_ConfusionMatrix.png"
                                .format(cwd, directory, algorithm, size_str, dataset))
                plt.close('all')

                print(classifier_score)
                count += 1
        
        for i in range(len(algorithms)):
            with plt.style.context('bmh'):
                fig, ax = plt.subplots()
                ax.set_xlabel("Percent of Training Set")
                ax.set_ylabel("Accuracy (%)")
                ax.set_title("Accuracy vs Training Set Size ({})")
                
                ax.plot([i for i in range(0,
                                          int((stop * percent) * 100) + 1,
                                          int((start / stop) * 100))],
                        accuracy[i], marker='o', label=algorithms[i])
                ax.legend()
                plt.savefig("{}/{}/{}_{}_Training_Size_Impact.png"
                            .format(cwd, directory, algorithms[i], dataset))
        
        accuracy_array = np.asarray(accuracy)
        BallTree_run_time_array = np.asarray(ball_tree_run_time)
        KdTree_run_time_array = np.asarray(kd_tree_run_time)
        Brute_run_time_array = np.asarray(brute_run_time)
        for i in range(len(accuracy_array)):
            accuracy_array.tofile("{}/{}/KNN_{}_Accuracy.csv"
                                  .format(cwd, directory, algorithms[i]), sep=",", format="%.3f")
        BallTree_run_time_array.tofile("{}/{}/KNN_{}_RunTime.csv"
                                       .format(cwd, directory, 'BallTree'), sep=",", format="%.3f")
        KdTree_run_time_array.tofile("{}/{}/KNN_{}_RunTime.csv"
                                     .format(cwd, directory, "KdTree"), sep=",", format="%.3f")

        Brute_run_time_array.tofile("{}/{}/KNN_{}_RunTime.csv"
                                    .format(cwd, directory, 'BruteForce'), sep=",", format="%.3f")

    except Exception as err:
        print("Exception occurred while comparing how training size impacts accuracy. \n", err)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def BoostedDecisionTreeTrainingSizeCompare(training_data, training_labels, testing_data, testing_labels, dataset):
    try:
        if training_data is None \
                or training_labels is None \
                or testing_data is None \
                or testing_labels is None:
            return
        """
        index 0 = 25% of training set
        index 1 = 50% of training set
        index 2 = 75% of training set
        index 3 = 100% of training set
        """
        cwd = pathlib.Path().absolute()
        directory = "Boosting_Results/Boosting_Images/"
        clf_type = 'BoostedDecisionTree'
        number_of_data_entries = training_data.shape[0]
        run_time = [0.0]
        accuracy = [0.0]
        sizes = [0.0]
        start = 1
        stop = Dataset_Splits
        percent = (start / stop)
        for i in range(start, stop + 1, 1):
            plt.close('all')
            size = i * percent
            sizes.append(int(size * 100))
            Classifier = GradientBoostingClassifier(n_estimators=200, max_depth=30, verbose=3)
            set_0 = {"Training_Data": training_data[:int(number_of_data_entries * size)],
                     "Training_Labels": training_labels[:int(number_of_data_entries * size)]}
            start_time = timer()
            with parallel_backend('threading'):
                Classifier.fit(set_0["Training_Data"], set_0["Training_Labels"])
            end_time = timer()
            elapsed_time = end_time - start_time
            run_time.append(elapsed_time)
            y_pred = Classifier.predict(testing_data)
            temp_confusion_matrix = confusion_matrix(testing_labels, y_pred)
            temp_classification_report = classification_report(testing_labels, y_pred)
            size_str = str(int(size * 100))
            print("Confusion Matrix \n {} \n".format(temp_confusion_matrix))
            print("Classification Report \n {} \n".format(temp_classification_report))
            classifier_score = Classifier.score(testing_data, testing_labels)
            accuracy.append(classifier_score)
            with plt.style.context('bmh'):
                disp = plot_confusion_matrix(Classifier, testing_data, testing_labels, values_format=".4g")
                disp.figure_.suptitle("Confusion Matrix {}% Training Data".format(size_str))
                print("TESTING {}%".format(size_str))
                print(disp.confusion_matrix)
                plt.savefig("{}//{}_{}_ConfusionMatrix_{}%_Training.png"
                            .format(cwd, directory, clf_type, dataset, size_str))
            print(classifier_score)

        with plt.style.context('bmh'):
            fig, ax = plt.subplots()
            ax.set_xlabel("Percent of Training Set")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Accuracy vs Training Set Size")

            ax.plot([i for i in range(0,
                                      int((stop * percent) * 100) + 1,
                                      int((start / stop) * 100))],
                    accuracy, marker='o', label="accuracy")
            ax.legend()
            plt.savefig("{}/{}/{}_{}_Training_Set_Size_Impact.png"
                        .format(cwd, directory, clf_type, dataset))

        with plt.style.context('ggplot'):
            fig0, ax0 = plt.subplots()
            ax0.set_xlabel("Percent of Training Set")
            ax0.set_ylabel("Accuracy (%)", color='tab:green')
            ax0.set_title("Accuracy vs Training Set Size")
            ax0.plot([i for i in range(0, int((stop * percent) * 100) + 1, int((start / stop) * 100))],
                     accuracy, "tab:green", marker='o', label="accuracy")
            ax0.tick_params(axis='y', labelcolor="tab:green")

            ax3 = ax0.twinx()
            ax3.set_ylabel("Training Time (s)", color="tab:blue")
            ax3.plot([i for i in range(0, int((stop * percent) * 100) + 1, int((start / stop) * 100))],
                     run_time, "tab:blue", marker='o', label="training-time")
            ax3.tick_params(axis='y', labelcolor="tab:blue")
            fig0.tight_layout()
            plt.savefig("{}/{}/{}_{}_Training_Set_Size_Impact_vs_Training_Time.png"
                        .format(cwd, directory, clf_type, dataset))

    except Exception as err:
        print("Exception occurred while comparing how training size impacts accuracy. \n", err)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def BoostedDecisionTreeDepthCompare(training_data, training_labels, testing_data,
                                    testing_labels, dataset, tree_max_depth=50):
    try:
        if training_data is None \
                or training_labels is None \
                or testing_data is None \
                or testing_labels is None:
            return
        """
        https://towardsdatascience.com/understanding-decision-trees-for-classification-python-9663d683c952
        by Michael Galarnyk
        """
        accuracy = [0.0]
        run_time = [0.0]
        cwd = pathlib.Path().absolute()

        for depth in range(1, tree_max_depth + 1):
            clf = GradientBoostingClassifier(n_estimators=200, max_depth=depth, verbose=3)
            start_time = timer()
            clf.fit(training_data, training_labels)
            end_time = timer()
            elapsed_time = end_time - start_time
            run_time.append(elapsed_time)
            score = clf.score(testing_data, testing_labels)
            print("Current Depth Tree {} / {} ".format(depth, tree_max_depth))
            print("Accuracy: {0:.3f}%\n".format(score))
            accuracy.append(score)
        results = np.asarray(accuracy)
        results.tofile("{}/DecisionTree_Results/DecisionTree_"
                       "{}_Depth_Compare.csv".format(cwd, dataset),
                       sep=",", format="%.4f")
        with plt.style.context('ggplot'):
            fig, ax = plt.subplots()
            ax.set_xlabel("Tree Depth")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Accuracy vs Tree Depth for Testing sets")
            ax.plot([i for i in range(tree_max_depth + 1)], accuracy, marker='o', label="test")
            ax.legend()
            plt.savefig("{}/Boosting_Results/Boosting_Images/Boosted_Tree_"
                        "{}_Tree_Depth_VS_Accuracy.png".format(cwd, dataset))

        with plt.style.context('ggplot'):
            fig1, ax1 = plt.subplots()
            ax1.set_xlabel("Tree Depth")
            ax1.set_ylabel("Accuracy (%)", color='tab:red')
            ax1.plot([i for i in range(tree_max_depth + 1)], accuracy, "tab:red", marker='o', label="accuracy")
            ax1.set_title("Accuracy vs Tree Depth vs Training Time")
            ax1.tick_params(axis="y", labelcolor="tab:red")

            ax2 = ax1.twinx()
            ax2.set_ylabel("Training Time (s)", color="tab:blue")
            ax2.plot([i for i in range(tree_max_depth + 1)], run_time, "tab:blue", marker='o', label="training-time")
            ax2.tick_params(axis="y", labelcolor="tab:blue")
            fig1.tight_layout()
            plt.savefig("{}/Boosting_Results/Boosting_Images/Boosted_Tree_"
                        "{}_Tree_Depth_VS_Accuracy_VS"
                        "_Training_Time.png".format(cwd, dataset))

        print()
    except Exception as err:
        print("Exception occurred while comparing Decision Tree depths impact on accuracy. \n", err)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


