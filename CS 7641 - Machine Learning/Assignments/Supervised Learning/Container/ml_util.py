import copy
import glob
import os
import pickle
import sys
import time

import joblib
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from joblib import delayed, Parallel, parallel_backend
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500

plt.tight_layout()

# From Hands on Machine Learning chapter 3 classification
# https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def setup(names=('MNIST',), train_set_size=10000, test_set_size=10000):
    try:
        MYDIR = "dataset"
        CHECK_FOLDER = os.path.isdir(MYDIR)
        
        # If folder doesn't exist, then create it.
        if not CHECK_FOLDER:
            os.makedirs(MYDIR)
            print("created folder : ", MYDIR)
        else:
            print(MYDIR, "folder already exists.")
        
        dataset_directory = f"{os.getcwd()}/dataset"
        
        files = glob.glob(f"{dataset_directory}/*.feather")
        file_names = set([i.split("\\")[-1]
                         .split(".")[0] for i in files])
        alt_file_names = set([i.split("/")[-1].split(".")[0] for i in files])
        dataset_path = "{}/dataset".format(os.getcwd())
        dataset = ""
        data = None
        results = {}
        
        for idx in range(len(names)):
            dataset = {"X": "", "y": "", "Full_Dataframe": ""}
            if names[idx].lower() not in file_names and names[idx].lower() not in alt_file_names:
                # if dataset not downloaded, download it
                print(f"\tDownloading {names[idx]} dataset")
                time.sleep(0.1)
                if names[idx].lower() == "mnist":
                    df = fetch_openml('mnist_784', version=1, as_frame=True)
                else:
                    df = fetch_openml('fashion-MNIST', version=1, as_frame=True)
                data = df.frame.astype("uint8")
                print(f"\n\tFinished downloading {names[idx]} dataset")
                print(f"\tSaving {names[idx]} dataset as: \n\t'../dataset/{names[idx].lower()}.feather'")
                data.to_feather(f"{dataset_path}/{names[idx].lower()}.feather")
                dataset["X"] = data.drop(columns=["class"])
                dataset["y"] = data["class"]
                dataset["Full_Dataframe"] = data
                results[names[idx]] = dataset
            else:
                # Load the dataset
                print(f"{names[idx]} dataset found:")
                print(f"\tLoading {names[idx]}.feather")
                time.sleep(0.1)
                data = pd.read_feather(f"{dataset_path}/{names[idx].lower()}.feather").astype("uint8")
                dataset["X"] = data.drop(columns=["class"])
                dataset["y"] = data["class"]
                dataset["Full_Dataframe"] = data
                results[names[idx]] = dataset
                print(f"\tFinished loading {names[idx]} dataset")
        return results
    except Exception as e:
        print(f"Exception in setup:\n", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def split_data(X, y, scale=False, normalize=True):
    try:
        if scale and not normalize:
            X = StandardScaler().fit_transform(X)
        if normalize and not scale:
            X /= max(X.max())
        
        train_X, test_X, train_y, test_y = train_test_split(X, y.to_numpy(), test_size=0.20, random_state=42)
        test_X, valid_X = np.split(test_X.astype(np.float32), 2)
        test_y, valid_y = np.split(test_y.astype(np.uint8), 2)
        return pd.DataFrame(train_X), pd.Series(train_y), pd.DataFrame(valid_X), \
               pd.Series(valid_y), pd.DataFrame(test_X), pd.Series(test_y)
    except Exception as e:
        print(f"Exception in split_data:\n", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    # From Hands on Machine Learning chapter 3 classification
    # https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb
    plt.close('all')
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution, bbox_inches='tight')
    return


def plot_digit(data):
    # From Hands on Machine Learning chapter 3 classification
    # https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb
    plt.close('all')
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    return


def plot_digits(instances, images_per_row=10, **options):
    # From Hands on Machine Learning chapter 3 classification
    # https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb
    try:
        plt.close('all')
        size = 28
        images_per_row = min(len(instances), images_per_row)
        images = [instance.reshape(size, size) for instance in instances]
        n_rows = (len(instances) - 1) // images_per_row + 1
        row_images = []
        n_empty = n_rows * images_per_row - len(instances)
        images.append(np.zeros((size, size * n_empty)))
        for row in range(n_rows):
            rimages = images[row * images_per_row: (row + 1) * images_per_row]
            row_images.append(np.concatenate(rimages, axis=1))
        image = np.concatenate(row_images, axis=0)
        plt.imshow(image, cmap=mpl.cm.binary, **options)
        plt.axis("off")
        return
    except Exception as e:
        print(f"Exception in plot_digits:\n", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def generate_image_grid(class_names, data_X, data_y, random=False, name="", save_dir=""):
    try:
        plt.close('all')
        if save_dir is not None and save_dir != "":
            CHECK_FOLDER = os.path.isdir(save_dir)
            
            # If folder doesn't exist, then create it.
            if not CHECK_FOLDER:
                os.makedirs(save_dir)
                print("created folder : ", save_dir)
            else:
                print(save_dir, "folder already exists.")
        else:
            save_dir = os.getcwd()
        if not random:
            val, idx = np.unique(data_y, return_index=True)
            print(val)
            print(idx)
            plt.figure(figsize=(10, 10))
            for i in range(val.shape[0]):
                plt.subplot(5, 5, i + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(data_X[idx[i]].reshape((28, 28)), cmap=plt.cm.binary)
                plt.xlabel(class_names[data_y[idx[i]]])
        else:
            plt.figure(figsize=(10, 10))
            for i in range(25):
                plt.subplot(5, 5, i + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(data_X[i].reshape((28, 28)), cmap=plt.cm.binary)
                plt.xlabel(class_names[data_y[i]])
        if not random:
            plt.savefig(f"{save_dir}/Image_Grid_{name}.png", bbox_inches='tight')
        else:
            plt.savefig(f"{save_dir}/Image_Grid_{name}_random.png", bbox_inches='tight')
        return
    except Exception as e:
        print(f"Exception in plot_digits:\n", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def combined_generate_image_grid(class_one_names, class_two_names, data1_X, data1_y, data2_X,
                                 data2_y, random=False, save_dir=""):
    try:
        _0, mnist_idx, _2 = np.unique(data1_y, return_index=True, return_counts=True)
        _1, fash_idx, _3 = np.unique(data2_y, return_index=True, return_counts=True)
        
        data1_X = data1_X[mnist_idx, :]
        data1_y = data1_y[mnist_idx]
        data2_X = data2_X[fash_idx, :]
        data2_y = data2_y[fash_idx]
        
        cols = 6
        plt.close('all')
        if save_dir is not None and save_dir != "":
            CHECK_FOLDER = os.path.isdir(save_dir)
            
            # If folder doesn't exist, then create it.
            if not CHECK_FOLDER:
                os.makedirs(save_dir)
                print("created folder : ", save_dir)
            else:
                print(save_dir, "folder already exists.")
        else:
            save_dir = os.getcwd()
        plt.close("all")
        
        fig = plt.figure(figsize=(20, 10))
        
        outer_grid = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.1)
        
        ax1 = plt.Subplot(fig, outer_grid[0, 0])
        ax2 = plt.Subplot(fig, outer_grid[0, 1])
        ax1.set_title("MNIST", fontsize=25, weight='bold')
        ax1.axis('off')
        ax2.set_title("Fashion-MNIST", fontsize=25, weight='bold')
        ax2.axis('off')
        fig.add_subplot(ax1)
        fig.add_subplot(ax2)
        
        inner_grid_1 = outer_grid[0, 0].subgridspec(3, 3, hspace=0.1, wspace=0.1)
        inner_grid_2 = outer_grid[0, 1].subgridspec(3, 3, hspace=0.1, wspace=0.1)
        
        for g in range(2):
            if g == 0:
                names = class_one_names
                data = data1_X
                labels = data1_y
                gd = inner_grid_1
            else:
                names = class_two_names
                data = data2_X
                labels = data2_y
                gd = inner_grid_2
            
            for i in range(9):

                ax = fig.add_subplot(gd[i], autoscale_on=True)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(data[i].reshape((28, 28)), cmap=plt.cm.binary)
                plt.xlabel(names[labels[i]], fontsize=20)
                fig.add_subplot(ax)
        
        plt.savefig(f"{save_dir}/Image_Grid_Combined_random.png")
        return
    except Exception as e:
        print(f"Exception in combined_generate_image_grid:\n", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def plot_learning_curve(estimator, title, train_X, train_y, axes=None, ylim=(0.6, 1.01), cv=None, f_name="My_Plot",
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), save_individual=False,
                        TESTING=False, backend='loky', extra_name="", folder="SVM"):
    """
    FROM https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    :param backend:
    :param estimator: sklearn classifier
    :param title: string - Title for the plot
    :param train_X: numpy array - training train_X
    :param train_y: numpy array - training y
    :param axes: matplotlib axes
    :param ylim: tuple - limit for the y axes
    :param cv: integer - number of cross-validation
    :param f_name: string - filename
    :param n_jobs: integer - number of concurrent processes
    :param train_sizes: numpy array - a numpy array of percentages from 0.0 to 1.0 which will be used for training
    :param save_individual: boolean - boolean value to determine if we are saving individual copies of charts
    :param TESTING: boolean - True/False if testing or not to determine verbose
    :return: plt: object,
            temp_df: pandas dataframe,
            results: dictionary of all crossvalidation results
    """
    try:
        time.sleep(5)
        plt.close('all')
        if TESTING:
            verbose = True
            verbose_val = 10
        else:
            verbose = False
            verbose_val = 0
        
        with joblib.parallel_backend(backend=backend, n_jobs=n_jobs):
            train_sizes, train_scores, test_scores, fit_times, eval_times = \
                learning_curve(estimator, train_X, train_y, train_sizes=train_sizes, cv=cv, n_jobs=n_jobs,
                               random_state=42,
                               return_times=True, verbose=verbose_val)
        
        temp_data = {"Training Scores": np.mean(train_scores, axis=1),
                     "Testing Scores": np.mean(test_scores, axis=1),
                     "Fit Times": np.mean(fit_times, axis=1),
                     "Evaluation Times": np.mean(eval_times, axis=1)}
        temp_df = pd.DataFrame(data=temp_data, index=train_sizes)
        
        cols = [f"CV-{i}" for i in range(cv)]
        all_train_scores = pd.DataFrame(data=train_scores, columns=cols, index=train_sizes)
        all_test_scores = pd.DataFrame(data=test_scores, columns=cols, index=train_sizes)
        all_fit_times = pd.DataFrame(data=fit_times, columns=cols, index=train_sizes)
        all_eval_times = pd.DataFrame(data=eval_times, columns=cols, index=train_sizes)
        results = {"Training Scores": all_train_scores,
                   "Testing Scores": all_test_scores,
                   "Fit Times": all_fit_times,
                   "Evaluation Times": all_eval_times}
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
        
        if save_individual:
            plt.close("all")
            plt.grid()
            plt.ylim(ylim[0], ylim[1])
            plt.grid(which='major', linestyle='-', linewidth='0.5', color='white')
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.2,
                             color="darkorange")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.2,
                             color="navy")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="navy",
                     label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="darkorange",
                     label="Cross-validation score")
            plt.title(title, fontsize=15, weight='bold')
            plt.xlabel("Training examples", fontsize=15, weight='heavy')
            plt.ylabel("Score", fontsize=15, weight='heavy')
            plt.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
            plt.tight_layout()
            plt.savefig(f"{os.getcwd()}/figures/{folder}/Learning_Curves/{f_name}_{extra_name}_Learning_Curve.png",
                        bbox_inches='tight')
            
            plt.close("all")
            plt.grid()
            plt.grid(which='major', linestyle='-', linewidth='0.5', color='white')
            plt.plot(train_sizes, fit_times_mean, 'o-', label='Fit Times', color="darkorange")
            plt.fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.2,
                             color="darkorange")
            plt.xlabel("Training examples", fontsize=15, weight='heavy')
            plt.ylabel("fit_times", fontsize=15, weight='heavy')
            plt.title("Scalability of the model", fontsize=15, weight='bold')
            plt.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
            plt.tight_layout()
            plt.savefig(f"{os.getcwd()}/figures/{folder}/Learning_Curves/{f_name}_{extra_name}_Fit_Times.png",
                        bbox_inches='tight')
            
            plt.close("all")
            plt.ylim(ylim[0], ylim[1])
            plt.grid()
            plt.grid(which='major', linestyle='-', linewidth='0.5', color='white')
            plt.plot(fit_times_mean, test_scores_mean, 'o-', label="Scalability", color="darkorange")
            plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.2,
                             color="darkorange")
            plt.xlabel("Training examples", fontsize=15, weight='heavy')
            plt.ylabel("Score", fontsize=15, weight='heavy')
            plt.title("Performance of the model", fontsize=15, weight='bold')
            plt.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
            plt.tight_layout()
            plt.savefig(f"{os.getcwd()}/figures/{folder}/Learning_Curves/{f_name}_{extra_name}_Fit_Times_Vs_Score.png",
                        bbox_inches='tight')
        
        plt.close('all')
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        axes[0].set_title(title, fontsize=15, weight='bold')
        plt.ylim(ylim[0], ylim[1])
        ylim = (0.6, 1.01)
        axes[0].set_xlabel("Training examples", fontsize=15, weight='heavy')
        axes[0].set_ylabel("Score", fontsize=15, weight='heavy')
        axes[0].set_ylim(ylim[0], ylim[1])
        
        # Plot learning curve
        axes[0].grid()
        # Customize the major grid
        axes[0].grid(which='major', linestyle='-', linewidth='0.5', color='white')
        
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.2,
                             color="navy")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.2,
                             color="darkorange")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="navy",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="darkorange",
                     label="Cross-validation score")
        axes[0].set_ylim(ylim[0], ylim[1])
        axes[0].legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        
        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].grid(which='major', linestyle='-', linewidth='0.5', color='white')
        axes[1].plot(train_sizes, fit_times_mean, 'o-', label="Fit Times", color="darkorange")
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.2,
                             color="darkorange")
        axes[1].set_xlabel("Training examples", fontsize=15, weight='heavy')
        axes[1].set_ylabel("fit_times", fontsize=15, weight='heavy')
        axes[1].set_title("Scalability of the model", fontsize=15, weight='bold')
        axes[1].legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        
        # Plot fit_time vs score
        axes[2].grid()
        axes[2].grid(which='major', linestyle='-', linewidth='0.5', color='white')
        axes[2].plot(train_sizes, test_scores_mean, 'o-', label="Scalability", color="darkorange")
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.2,
                             color="darkorange")
        axes[2].set_xlabel("Training examples", fontsize=15, weight='heavy')
        axes[2].set_ylabel("Score", fontsize=15, weight='heavy')
        axes[2].set_ylim(ylim[0], ylim[1])
        axes[2].set_title("Performance of the model", fontsize=15, weight='bold')
        axes[2].legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/figures/{folder}/Learning_Curves/{f_name}_{extra_name}.png", bbox_inches='tight')
        return temp_df, results
    
    except Exception as e:
        print(f"Exception in plot_learning_curve:\n", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def get_cv_result(classifier, train_X, train_y, valid_X, valid_y, test_X, test_y):
    start_time = time.time()
    classifier.fit(train_X, train_y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    y_pred_train = classifier.predict(train_X)
    y_pred_valid = classifier.predict(valid_X)
    y_pred_test = classifier.predict(test_X)
    results = {"Results": None,
               "Accuracy": None,
               "Run Time": elapsed_time}
    
    temp_results = {"Training": {"Precision": np.zeros(shape=(1,)),
                                 "Recall": np.zeros(shape=(1,)),
                                 "F1": np.zeros(shape=(1,))},
                    "Validation": {"Precision": np.zeros(shape=(1,)),
                                   "Recall": np.zeros(shape=(1,)),
                                   "F1": np.zeros(shape=(1,))},
                    "Testing": {"Precision": np.zeros(shape=(1,)),
                                "Recall": np.zeros(shape=(1,)),
                                "F1": np.zeros(shape=(1,))}}
    acc = {
        "Training": None,
        "Validation": None,
        "Testing": None
    }
    
    temp_results["Training"]["Precision"], \
    temp_results["Training"]["Recall"], \
    temp_results["Training"]["F1"], _ = precision_recall_fscore_support(y_true=train_y, y_pred=y_pred_train,
                                                                        average="weighted")
    
    temp_results["Validation"]["Precision"], \
    temp_results["Validation"]["Recall"], \
    temp_results["Validation"]["F1"], _ = precision_recall_fscore_support(y_true=valid_y, y_pred=y_pred_valid,
                                                                          average="weighted")
    
    temp_results["Testing"]["Precision"], \
    temp_results["Testing"]["Recall"], \
    temp_results["Testing"]["F1"], _ = precision_recall_fscore_support(y_true=test_y, y_pred=y_pred_test,
                                                                       average="weighted")
    
    acc["Training"] = accuracy_score(y_true=train_y, y_pred=y_pred_train)
    acc["Validation"] = accuracy_score(y_true=valid_y, y_pred=y_pred_valid)
    acc["Testing"] = accuracy_score(y_true=test_y, y_pred=y_pred_test)
    results["Results"] = temp_results
    results["Accuracy"] = acc
    return results


def evaluate_learner(classifier, train_set, test_set, validation_set, idx=np.zeros(shape=(1,)), cv=5, is_nn=False):
    data = {"Training Accuracy": np.zeros(shape=(idx.shape[0],)),
            "Testing Accuracy": np.zeros(shape=(idx.shape[0],)),
            "Training Time": np.zeros(shape=(idx.shape[0],)),
            "Training Precision": np.zeros(shape=(idx.shape[0],)),
            "Training Recall": np.zeros(shape=(idx.shape[0],)),
            "Training F1": np.zeros(shape=(idx.shape[0],)),
            "Validation Precision": np.zeros(shape=(idx.shape[0],)),
            "Validation Recall": np.zeros(shape=(idx.shape[0],)),
            "Validation F1": np.zeros(shape=(idx.shape[0],)),
            "Testing Precision": np.zeros(shape=(idx.shape[0],)),
            "Testing Recall": np.zeros(shape=(idx.shape[0],)),
            "Testing F1": np.zeros(shape=(idx.shape[0],))}
    all_cv_results = {}
    
    #
    train_X = train_set["X"]
    train_y = train_set["y"]
    test_X = test_set["X"]
    test_y = test_set["y"]
    valid_X = validation_set["X"]
    valid_y = validation_set["y"]
    
    starting_time = time.time()
    count = 0
    for i in idx:
        
        # region Stuff
        print(f"Current Training Set Size: {i}")
        training_subsets = np.random.choice(np.arange(train_X.shape[0]), size=(cv, i), replace=True)
        testing_idx_for_training = np.random.choice(np.arange(train_X.shape[0]), 1000, replace=True)
        testing_idx_for_valid_and_testing = np.random.choice(np.arange(test_X.shape[0]), 1000, replace=True)
        
        print(f"Beginning Cross Validation:")
        
        classifiers = [copy.copy(classifier) for i in range(cv)]
        
        train_set_x = [train_X.iloc[training_subsets[j], :].to_numpy() for j in range(cv)]
        train_set_y = [train_y.iloc[training_subsets[j]].to_numpy() for j in range(cv)]
        
        valid_set_x = [valid_X.iloc[testing_idx_for_valid_and_testing, :] for j in range(cv)]
        valid_set_y = [valid_y.iloc[testing_idx_for_valid_and_testing] for j in range(cv)]
        
        test_set_x = [test_X.iloc[testing_idx_for_valid_and_testing, :] for j in range(cv)]
        test_set_y = [test_y.iloc[testing_idx_for_valid_and_testing] for j in range(cv)]
        cv_time_results = np.zeros(shape=(cv,))
        
        cv_train_precision = np.zeros(shape=(cv,))
        cv_train_recall = np.zeros(shape=(cv,))
        cv_train_f1 = np.zeros(shape=(cv,))
        cv_train_acc = np.zeros(shape=(cv,))
        
        cv_valid_precision = np.zeros(shape=(cv,))
        cv_valid_recall = np.zeros(shape=(cv,))
        cv_valid_f1 = np.zeros(shape=(cv,))
        cv_valid_acc = np.zeros(shape=(cv,))
        
        cv_test_precision = np.zeros(shape=(cv,))
        cv_test_recall = np.zeros(shape=(cv,))
        cv_test_f1 = np.zeros(shape=(cv,))
        cv_test_acc = np.zeros(shape=(cv,))
        # endregion
        
        cv_start_time = time.time()
        res = Parallel(n_jobs=cv, backend="threading", verbose=1)(delayed(
            get_cv_result)(classifier=classifiers[i], train_X=train_set_x[i], train_y=train_set_y[i],
                           valid_X=valid_set_x[i], valid_y=valid_set_y[i], test_X=test_set_x[i],
                           test_y=test_set_y[i]) for i in range(cv))
        
        cv_end_time = time.time()
        cv_elapsed_time = cv_end_time - cv_start_time
        print(f"Cross-Validation Time: {cv_elapsed_time:.6f}s")
        
        for t_idx in range(cv):
            for key_a, val_a in res[t_idx]["Results"].items():
                if key_a == "Training":
                    for key_b, val_b in val_a.items():
                        if key_b == "Precision":
                            cv_train_precision[t_idx] = val_b
                        if key_b == "Recall":
                            cv_train_recall[t_idx] = val_b
                        if key_b == "F1":
                            cv_train_f1[t_idx] = val_b
                
                if key_a == "Validation":
                    for key_b, val_b in val_a.items():
                        if key_b == "Precision":
                            cv_valid_precision[t_idx] = val_b
                        if key_b == "Recall":
                            cv_valid_recall[t_idx] = val_b
                        if key_b == "F1":
                            cv_valid_f1[t_idx] = val_b
                
                if key_a == "Testing":
                    for key_b, val_b in val_a.items():
                        if key_b == "Precision":
                            cv_test_precision[t_idx] = val_b
                        if key_b == "Recall":
                            cv_test_recall[t_idx] = val_b
                        if key_b == "F1":
                            cv_test_f1[t_idx] = val_b
            for key_a, val_b in res[t_idx]["Accuracy"].items():
                if key_a == "Training":
                    cv_train_acc[t_idx] = val_b
                if key_a == "Validation":
                    cv_valid_acc[t_idx] = val_b
                if key_a == "Testing":
                    cv_test_acc[t_idx] = val_b
            cv_time_results[t_idx] = res[t_idx]["Run Time"]
        data["Training Accuracy"][count] = np.mean(cv_train_acc)
        data["Testing Accuracy"][count] = np.mean(cv_test_acc)
        data["Training Time"][count] = np.mean(cv_time_results)
        data["Training Precision"][count] = np.mean(cv_train_precision)
        data["Training Recall"][count] = np.mean(cv_train_recall)
        data["Training F1"][count] = np.mean(cv_train_f1)
        data["Validation Precision"][count] = np.mean(cv_valid_precision)
        data["Validation Recall"][count] = np.mean(cv_valid_recall)
        data["Validation F1"][count] = np.mean(cv_valid_f1)
        data["Testing Precision"][count] = np.mean(cv_test_precision)
        data["Testing Recall"][count] = np.mean(cv_test_recall)
        data["Testing F1"][count] = np.mean(cv_test_f1)
        count += 1
    
    end_t = time.time()
    elapsed = end_t - starting_time
    print(f"Parallel Time: {elapsed}")
    print("Finished")
    results = pd.DataFrame(data=data, index=idx)
    return results


def run_grid_search(classifier, parameters, train_X, train_y, cv=5, n_jobs=-1, verbose=1, return_train_score=False,
                    refit=True, save_dir="figures/", algorithm_name="DT", backend='threading', extra_f_name="",
                    extra_name="", folder="SVM"):
    try:
        with parallel_backend(backend=backend, n_jobs=n_jobs):
            grid = GridSearchCV(estimator=classifier, param_grid=parameters, n_jobs=n_jobs,
                                cv=cv, verbose=verbose, return_train_score=return_train_score, refit=refit)
            grid.fit(X=train_X, y=train_y)
            temp_df = pd.DataFrame(data=grid.cv_results_)
            temp_df.to_pickle(f"{save_dir}/Grid_Search_Results/{algorithm_name}_Grid_Search_Results.pkl")
            
            pkl_filename = f"{save_dir}/Grid_Search_Results/{algorithm_name}_{extra_f_name}Grid_Object_{extra_name}_.pkl"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(grid, file)
        
        return temp_df, grid
    except Exception as e:
        print(f"Exception in run_grid_search:\n", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def get_learning_curve(classifier, train_X, train_y, train_sizes=np.linspace(0.1, 1.0, 5), cv=5,
                       n_jobs=-1, verbose=1, random_number=42, return_times=False, backend='threading',
                       extra_name="", folder_name=""):
    try:
        with parallel_backend(backend=backend, n_jobs=n_jobs):
            train_sizes, train_scores, test_scores, \
            fit_times, eval_times = learning_curve(classifier, train_X, train_y,
                                                   train_sizes=train_sizes, cv=cv, verbose=verbose,
                                                   random_state=random_number, return_times=return_times)
        
        temp_data = {"Training Scores": np.mean(train_scores, axis=1),
                     "Testing Scores": np.mean(test_scores, axis=1),
                     "Fit Times": np.mean(fit_times, axis=1),
                     "Evaluation Times": np.mean(eval_times, axis=1)}
        temp_df = pd.DataFrame(data=temp_data, index=train_sizes)
        cols = [f"CV-{i}" for i in range(cv)]
        
        all_train_scores = pd.DataFrame(data=train_scores, columns=cols, index=train_sizes)
        all_test_scores = pd.DataFrame(data=test_scores, columns=cols, index=train_sizes)
        all_fit_times = pd.DataFrame(data=fit_times, columns=cols, index=train_sizes)
        all_eval_times = pd.DataFrame(data=eval_times, columns=cols, index=train_sizes)
        results = {"Training Scores": all_train_scores,
                   "Testing Scores": all_test_scores,
                   "Fit Times": all_fit_times,
                   "Evaluation Times": all_eval_times}
        return temp_df, results
    except Exception as e:
        print(f"Exception in get_learning_curve:\n", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def get_model_complexity(classifier, train_X, train_y, parameter_name, parameter_range, cv=5, n_jobs=-1, verbose=1,
                         backend='threading', algorithm_name="Model", save_dir="figures/", is_NN=False,
                         nn_range=None, param_name_for_plot="N", f_name="N", plot_title="N", is_SVM=False,
                         extra_name="", folder="SVM", use_log_x=False, fileName=""):
    try:
        plt.close('all')
        plt.style.use('ggplot')
        # Validation curve = Complexity curve. Compare the Bias and Variance
        with parallel_backend(backend=backend, n_jobs=n_jobs):
            train_scores, test_scores = validation_curve(classifier, train_X, train_y, param_name=parameter_name,
                                                         param_range=parameter_range, cv=cv, scoring="accuracy",
                                                         n_jobs=n_jobs, verbose=verbose, error_score=0.0)
        cols = [f"CV{i}" for i in range(cv)]
        train_data = {f"CV{i}": train_scores[:, i] for i in range(cv)}
        test_data = {f"CV{i}": test_scores[:, i] for i in range(cv)}
        parameter_range_column = {f"{parameter_name}": parameter_range}
        train_data.update(parameter_range_column)
        test_data.update(parameter_range_column)
        
        temp_train_df = pd.DataFrame(data=train_data)
        temp_test_df = pd.DataFrame(data=test_data)
        
        temp_train_df.to_pickle(
            f"{save_dir}/Complexity_Analysis/_{parameter_name}_Complexity_Analysis_Training_{extra_name}_.pkl")
        temp_test_df.to_pickle(
            f"{save_dir}/Complexity_Analysis/_{parameter_name}_Complexity_Analysis_Testing_{extra_name}_.pkl")
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        if param_name_for_plot == "N":
            param_name_for_plot = parameter_name
        
        if is_NN:
            parameter_range = nn_range
        
        plt.title(f"{plot_title}", weight='bold')
        plt.xlabel(f"{param_name_for_plot}", fontsize=15, weight='heavy')
        plt.ylabel("Accuracy", fontsize=15, weight='heavy')
        plt.ylim(0.4, 1.05)
        lw = 2
        
        if use_log_x:
            plt.semilogx(parameter_range, train_scores_mean, label="Training score",
                         color="darkorange", lw=lw)
            plt.semilogx(parameter_range, test_scores_mean, label="Cross-validation score",
                         color="navy", lw=lw)
        else:
            plt.plot(parameter_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
            plt.plot(parameter_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(parameter_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.fill_between(parameter_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.tight_layout()
        plt.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.savefig(f"{save_dir}/Complexity_Analysis/_{fileName}_.png",
                    bbox_inches='tight')
        return temp_train_df, temp_test_df
    except Exception as e:
        print(f"Exception in get_model_complexity:\n", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def plot_combined_complexity(model_name, x_label, parameter_range=None, is_NN=False, is_SVM=False,
                             plt_width=8, plt_height=6, orientation="Vertical", ylim=(0.4, 1.05),
                             svm_kernel_names=["RBF", "Linear"], only_one=False, which_one="RBF", extra_name="",
                             is_final=False, mnist_train_complex=None, mnist_test_complex=None,
                             fashion_train_complex=None, fashion_test_complex=None, folder="SVM",
                             alt_mnist_train_complex=None, alt_mnist_test_complex=None,
                             alt_fashion_train_complex=None, alt_fashion_test_complex=None, use_saved=False,
                             use_log_x=False):
    try:
        plt.style.use("ggplot")
        plt.close("all")
        mpl.rcParams['figure.figsize'] = [plt_width, plt_height]
        dataset_directory = f"{os.getcwd()}/figures/{model_name}"
        
        files = glob.glob(f"{dataset_directory}/*.pkl")
        
        mnist_complexity = {"Training": None,
                            "Testing": None}
        fashion_complexity = {"Training": None,
                              "Testing": None}
        
        kernel_0 = {'mnist_complexity': {"Training": None, "Testing": None},
                    'fashion_complexity': {"Training": None, "Testing": None}}
        kernel_1 = {'mnist_complexity': {"Training": None, "Testing": None},
                    'fashion_complexity': {"Training": None, "Testing": None}}
        if only_one:
            svm_kernel_names = [which_one, which_one]
        for i in files:
            if "Complexity" in i:
                if is_SVM:
                    if 'Fashion' in i or 'fashion' in i:
                        # Fashion Dataset
                        if svm_kernel_names[0] in i or svm_kernel_names[0].lower() in i:
                            if "Testing" in i or "testing" in i:
                                kernel_0['fashion_complexity']["Testing"] = i
                            else:
                                kernel_0['fashion_complexity']["Training"] = i
                        else:
                            if "Testing" in i or "testing" in i:
                                kernel_1['fashion_complexity']["Testing"] = i
                            else:
                                kernel_1['fashion_complexity']["Training"] = i
                    else:
                        if svm_kernel_names[0] in i or svm_kernel_names[0].lower() in i:
                            if "Testing" in i or "testing" in i:
                                kernel_0['mnist_complexity']["Testing"] = i
                            else:
                                kernel_0['mnist_complexity']["Training"] = i
                        else:
                            if "Testing" in i or "testing" in i:
                                kernel_1['mnist_complexity']["Testing"] = i
                            else:
                                kernel_1['mnist_complexity']["Training"] = i
                else:
                    if "Fashion" in i:
                        if "Testing" in i:
                            fashion_complexity["Testing"] = i
                        else:
                            fashion_complexity["Training"] = i
                    else:
                        if "Testing" in i:
                            mnist_complexity["Testing"] = i
                        else:
                            mnist_complexity["Training"] = i
        
        if not is_SVM:
            if use_saved or mnist_train_complex is None:
                mnist_train = pd.read_pickle(mnist_complexity["Training"])
                mnist_test = pd.read_pickle(mnist_complexity["Testing"])
                fashion_train = pd.read_pickle(fashion_complexity["Training"])
                fashion_test = pd.read_pickle(fashion_complexity["Testing"])
            else:
                mnist_train = mnist_train_complex
                mnist_test = mnist_test_complex
                fashion_train = fashion_train_complex
                fashion_test = fashion_test_complex
            mnist_train = mnist_train.iloc[:, :-1]
            mnist_test = mnist_test.iloc[:, :-1]
            fashion_train = fashion_train.iloc[:, :-1]
            fashion_test = fashion_test.iloc[:, :-1]
            
            parameter_column = mnist_train.iloc[:, -1].to_numpy().flatten()
            
            if parameter_range is None:
                parameter_range = parameter_column
            
            mnist_train_mean = np.mean(mnist_train, axis=1)
            mnist_train_std = np.std(mnist_train, axis=1)
            mnist_test_mean = np.mean(mnist_test, axis=1)
            mnist_test_std = np.std(mnist_test, axis=1)
            
            fashion_train_mean = np.mean(fashion_train, axis=1)
            fashion_train_std = np.std(fashion_train, axis=1)
            fashion_test_mean = np.mean(fashion_test, axis=1)
            fashion_test_std = np.std(fashion_test, axis=1)
            
            if orientation.lower() == "vertical":
                fig, (ax1, ax2) = plt.subplots(2)
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2)
            lw = 2
            
            # plt.title(f"{plot_title}", weight='bold')
            ax1.set_title("MNIST Model Complexity", weight='bold')
            ax1.set_xlabel(xlabel=x_label, fontsize=15, weight='heavy')
            ax1.set_ylabel(ylabel="Accuracy", fontsize=15, weight='heavy')
            ax1.set_ylim(ylim[0], ylim[1])
            
            if use_log_x:
                ax1.semilogx(parameter_range, mnist_train_mean, label="Training score",
                             color="darkorange", lw=lw)
                ax1.semilogx(parameter_range, mnist_test_mean, label="Cross-validation score",
                             color="navy", lw=lw)
            else:
                ax1.plot(parameter_range, mnist_train_mean, label="Training score",
                         color="darkorange", lw=lw)
                ax1.plot(parameter_range, mnist_test_mean, label="Cross-validation score",
                         color="navy", lw=lw)
            
            ax1.fill_between(parameter_range, mnist_train_mean - mnist_train_std,
                             mnist_train_mean + mnist_train_std, alpha=0.2,
                             color="darkorange", lw=lw)
            ax1.fill_between(parameter_range, mnist_test_mean - mnist_test_std,
                             mnist_test_mean + mnist_test_std, alpha=0.2,
                             color="navy", lw=lw)
            ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
            
            ax2.set_title("Fashion MNIST Model Complexity", weight='bold')
            ax2.set_xlabel(xlabel=x_label, fontsize=15, weight='heavy')
            ax2.set_ylabel(ylabel="Accuracy", fontsize=15, weight='heavy')
            ax2.set_ylim(ylim[0], ylim[1])
            
            if use_log_x:
                ax2.semilogx(parameter_range, fashion_train_mean, label="Training score",
                             color="darkorange", lw=lw)
                ax2.semilogx(parameter_range, fashion_test_mean, label="Cross-validation score",
                             color="navy", lw=lw)
            else:
                ax2.plot(parameter_range, fashion_train_mean, label="Training score",
                         color="darkorange", lw=lw)
                ax2.plot(parameter_range, fashion_test_mean, label="Cross-validation score",
                         color="navy", lw=lw)
            
            ax2.fill_between(parameter_range, fashion_train_mean - fashion_train_std,
                             fashion_train_mean + fashion_train_std, alpha=0.2,
                             color="darkorange", lw=lw)
            ax2.fill_between(parameter_range, fashion_test_mean - fashion_test_std,
                             fashion_test_mean + fashion_test_std, alpha=0.2,
                             color="navy", lw=lw)
            ax2.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
            
            plt.tight_layout()
            
            plt.savefig(f"{dataset_directory}/Complexity_Analysis/_{model_name}_{orientation}_{extra_name}_.png",
                        bbox_inches='tight')
        else:
            plt.close("all")
            if only_one:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                lw = 2
                if use_saved or mnist_train_complex is None:
                    kernel_0_mnist_train = pd.read_pickle(kernel_0['mnist_complexity']["Training"])
                    kernel_0_mnist_test = pd.read_pickle(kernel_0['mnist_complexity']["Testing"])
                    kernel_0_fashion_train = pd.read_pickle(kernel_0['fashion_complexity']["Training"])
                    kernel_0_fashion_test = pd.read_pickle(kernel_0['fashion_complexity']["Testing"])
                
                else:
                    kernel_0_mnist_train = mnist_train_complex
                    kernel_0_mnist_test = mnist_test_complex
                    kernel_0_fashion_train = fashion_train_complex
                    kernel_0_fashion_test = fashion_test_complex
                
                parameter_column = kernel_0_mnist_train.iloc[:, -1].to_numpy().flatten()
                if is_NN:
                    parameter_column = np.asarray([i[0] for i in parameter_column])
                
                if parameter_range is None:
                    parameter_range = parameter_column
                
                # region Kernel 0
                kernel_0_mnist_train = kernel_0_mnist_train.iloc[:, :-1]
                kernel_0_mnist_test = kernel_0_mnist_test.iloc[:, :-1]
                kernel_0_fashion_train = kernel_0_fashion_train.iloc[:, :-1]
                kernel_0_fashion_test = kernel_0_fashion_test.iloc[:, :-1]
                
                kernel_0_mnist_train_mean = np.mean(kernel_0_mnist_train, axis=1)
                kernel_0_mnist_train_std = np.std(kernel_0_mnist_train, axis=1)
                kernel_0_mnist_test_mean = np.mean(kernel_0_mnist_test, axis=1)
                kernel_0_mnist_test_std = np.std(kernel_0_mnist_test, axis=1)
                
                kernel_0_fashion_train_mean = np.mean(kernel_0_fashion_train, axis=1)
                kernel_0_fashion_train_std = np.std(kernel_0_fashion_train, axis=1)
                kernel_0_fashion_test_mean = np.mean(kernel_0_fashion_test, axis=1)
                kernel_0_fashion_test_std = np.std(kernel_0_fashion_test, axis=1)
                
                ax1.set_title(f"MNIST Model Complexity\nKernel:{svm_kernel_names[0]}", weight='bold')
                ax1.set_xlabel(xlabel=x_label, fontsize=15, weight='heavy')
                ax1.set_ylabel(ylabel="Accuracy", fontsize=15, weight='heavy')
                ax1.set_ylim(ylim[0], ylim[1])
                ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
                
                ax1.semilogx(parameter_range, kernel_0_mnist_train_mean, label="Training score",
                             color="darkorange", lw=lw)
                ax1.fill_between(parameter_range, kernel_0_mnist_train_mean - kernel_0_mnist_train_std,
                                 kernel_0_mnist_train_mean + kernel_0_mnist_train_std, alpha=0.2,
                                 color="darkorange", lw=lw)
                ax1.semilogx(parameter_range, kernel_0_mnist_test_mean, label="Cross-validation score",
                             color="navy", lw=lw)
                ax1.fill_between(parameter_range, kernel_0_mnist_test_mean - kernel_0_mnist_test_std,
                                 kernel_0_mnist_test_mean + kernel_0_mnist_test_std, alpha=0.2,
                                 color="navy", lw=lw)
                ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
                
                ax2.set_title(f"Fashion MNIST Model Complexity\nKernel:{svm_kernel_names[0]}", weight='bold')
                ax2.set_xlabel(xlabel=x_label, fontsize=15, weight='heavy')
                ax2.set_ylabel(ylabel="Accuracy", fontsize=15, weight='heavy')
                ax2.set_ylim(ylim[0], ylim[1])
                ax2.grid(which='major', linestyle='-', linewidth='0.5', color='white')
                
                ax2.semilogx(parameter_range, kernel_0_fashion_train_mean, label="Training score",
                             color="darkorange", lw=lw)
                ax2.fill_between(parameter_range, kernel_0_fashion_train_mean - kernel_0_fashion_train_std,
                                 kernel_0_fashion_train_mean + kernel_0_fashion_train_std, alpha=0.2,
                                 color="darkorange", lw=lw)
                ax2.semilogx(parameter_range, kernel_0_fashion_test_mean, label="Cross-validation score",
                             color="navy", lw=lw)
                ax2.fill_between(parameter_range, kernel_0_fashion_test_mean - kernel_0_fashion_test_std,
                                 kernel_0_fashion_test_mean + kernel_0_fashion_test_std, alpha=0.2,
                                 color="navy", lw=lw)
                ax2.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
                
                plt.tight_layout()
            else:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                lw = 2
                if use_saved or mnist_train_complex is None:
                    kernel_0_mnist_train = pd.read_pickle(kernel_0['mnist_complexity']["Training"])
                    kernel_0_mnist_test = pd.read_pickle(kernel_0['mnist_complexity']["Testing"])
                    kernel_0_fashion_train = pd.read_pickle(kernel_0['fashion_complexity']["Training"])
                    kernel_0_fashion_test = pd.read_pickle(kernel_0['fashion_complexity']["Testing"])
                    
                    kernel_1_mnist_train = pd.read_pickle(kernel_1['mnist_complexity']["Training"])
                    kernel_1_mnist_test = pd.read_pickle(kernel_1['mnist_complexity']["Testing"])
                    kernel_1_fashion_train = pd.read_pickle(kernel_1['fashion_complexity']["Training"])
                    kernel_1_fashion_test = pd.read_pickle(kernel_1['fashion_complexity']["Testing"])
                else:
                    kernel_0_mnist_train = mnist_train_complex
                    kernel_0_mnist_test = mnist_test_complex
                    kernel_0_fashion_train = fashion_train_complex
                    kernel_0_fashion_test = fashion_test_complex
                    
                    kernel_1_mnist_train = alt_mnist_train_complex
                    kernel_1_mnist_test = alt_mnist_test_complex
                    kernel_1_fashion_train = alt_fashion_train_complex
                    kernel_1_fashion_test = alt_fashion_test_complex
                
                parameter_column = kernel_0_mnist_train.iloc[:, -1].to_numpy().flatten()
                if is_NN:
                    parameter_column = np.asarray([i[0] for i in parameter_column])
                
                if parameter_range is None:
                    parameter_range = parameter_column
                
                # region Kernel 0
                kernel_0_mnist_train = kernel_0_mnist_train.iloc[:, :-1]
                kernel_0_mnist_test = kernel_0_mnist_test.iloc[:, :-1]
                kernel_0_fashion_train = kernel_0_fashion_train.iloc[:, :-1]
                kernel_0_fashion_test = kernel_0_fashion_test.iloc[:, :-1]
                
                kernel_0_mnist_train_mean = np.mean(kernel_0_mnist_train, axis=1)
                kernel_0_mnist_train_std = np.std(kernel_0_mnist_train, axis=1)
                kernel_0_mnist_test_mean = np.mean(kernel_0_mnist_test, axis=1)
                kernel_0_mnist_test_std = np.std(kernel_0_mnist_test, axis=1)
                
                kernel_0_fashion_train_mean = np.mean(kernel_0_fashion_train, axis=1)
                kernel_0_fashion_train_std = np.std(kernel_0_fashion_train, axis=1)
                kernel_0_fashion_test_mean = np.mean(kernel_0_fashion_test, axis=1)
                kernel_0_fashion_test_std = np.std(kernel_0_fashion_test, axis=1)
                
                ax1.set_title(f"MNIST Model Complexity\nKernel:{svm_kernel_names[0]}", weight='bold')
                ax1.set_xlabel(xlabel=x_label, fontsize=15, weight='heavy')
                ax1.set_ylabel(ylabel="Accuracy", fontsize=15, weight='heavy')
                ax1.set_ylim(ylim[0], ylim[1])
                
                ax1.semilogx(parameter_range, kernel_0_mnist_train_mean, label="Training score",
                             color="darkorange", lw=lw)
                ax1.fill_between(parameter_range, kernel_0_mnist_train_mean - kernel_0_mnist_train_std,
                                 kernel_0_mnist_train_mean + kernel_0_mnist_train_std, alpha=0.2,
                                 color="darkorange", lw=lw)
                ax1.semilogx(parameter_range, kernel_0_mnist_test_mean, label="Cross-validation score",
                             color="navy", lw=lw)
                ax1.fill_between(parameter_range, kernel_0_mnist_test_mean - kernel_0_mnist_test_std,
                                 kernel_0_mnist_test_mean + kernel_0_mnist_test_std, alpha=0.2,
                                 color="navy", lw=lw)
                ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
                
                ax2.set_title(f"Fashion MNIST Model Complexity\nKernel:{svm_kernel_names[0]}", weight='bold')
                ax2.set_xlabel(xlabel=x_label, fontsize=15, weight='heavy')
                ax2.set_ylabel(ylabel="Accuracy", fontsize=15, weight='heavy')
                ax2.set_ylim(ylim[0], ylim[1])
                
                ax2.semilogx(parameter_range, kernel_0_fashion_train_mean, label="Training score",
                             color="darkorange", lw=lw)
                ax2.fill_between(parameter_range, kernel_0_fashion_train_mean - kernel_0_fashion_train_std,
                                 kernel_0_fashion_train_mean + kernel_0_fashion_train_std, alpha=0.2,
                                 color="darkorange", lw=lw)
                ax2.semilogx(parameter_range, kernel_0_fashion_test_mean, label="Cross-validation score",
                             color="navy", lw=lw)
                ax2.fill_between(parameter_range, kernel_0_fashion_test_mean - kernel_0_fashion_test_std,
                                 kernel_0_fashion_test_mean + kernel_0_fashion_test_std, alpha=0.2,
                                 color="navy", lw=lw)
                ax2.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
                
                plt.tight_layout()
                # endregion
                
                # region Kernel 1
                kernel_1_mnist_train = kernel_1_mnist_train.iloc[:, :-1]
                kernel_1_mnist_test = kernel_1_mnist_test.iloc[:, :-1]
                kernel_1_fashion_train = kernel_1_fashion_train.iloc[:, :-1]
                kernel_1_fashion_test = kernel_1_fashion_test.iloc[:, :-1]
                
                kernel_1_mnist_train_mean = np.mean(kernel_1_mnist_train, axis=1)
                kernel_1_mnist_train_std = np.std(kernel_1_mnist_train, axis=1)
                kernel_1_mnist_test_mean = np.mean(kernel_1_mnist_test, axis=1)
                kernel_1_mnist_test_std = np.std(kernel_1_mnist_test, axis=1)
                
                kernel_1_fashion_train_mean = np.mean(kernel_1_fashion_train, axis=1)
                kernel_1_fashion_train_std = np.std(kernel_1_fashion_train, axis=1)
                kernel_1_fashion_test_mean = np.mean(kernel_1_fashion_test, axis=1)
                kernel_1_fashion_test_std = np.std(kernel_1_fashion_test, axis=1)
                ax3.set_title(f"MNIST Model Complexity\nKernel:{svm_kernel_names[1]}", weight='bold')
                ax3.set_xlabel(xlabel=x_label, fontsize=15, weight='heavy')
                ax3.set_ylabel(ylabel="Accuracy", fontsize=15, weight='heavy')
                ax3.set_ylim(ylim[0], ylim[1])
                
                ax3.semilogx(parameter_range, kernel_1_mnist_train_mean, label="Training score",
                             color="darkorange", lw=lw)
                ax3.fill_between(parameter_range, kernel_1_mnist_train_mean - kernel_1_mnist_train_std,
                                 kernel_1_mnist_train_mean + kernel_1_mnist_train_std, alpha=0.2,
                                 color="darkorange", lw=lw)
                ax3.semilogx(parameter_range, kernel_1_mnist_test_mean, label="Cross-validation score",
                             color="navy", lw=lw)
                ax3.fill_between(parameter_range, kernel_1_mnist_test_mean - kernel_1_mnist_test_std,
                                 kernel_1_mnist_test_mean + kernel_1_mnist_test_std, alpha=0.2,
                                 color="navy", lw=lw)
                ax3.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
                
                ax4.set_title(f"Fashion MNIST Model Complexity\nKernel:{svm_kernel_names[1]}", weight='bold')
                ax4.set_xlabel(xlabel=x_label, fontsize=15, weight='heavy')
                ax4.set_ylabel(ylabel="Accuracy", fontsize=15, weight='heavy')
                ax4.set_ylim(ylim[0], ylim[1])
                
                ax4.semilogx(parameter_range, kernel_1_fashion_train_mean, label="Training score",
                             color="darkorange", lw=lw)
                ax4.fill_between(parameter_range, kernel_1_fashion_train_mean - kernel_1_fashion_train_std,
                                 kernel_1_fashion_train_mean + kernel_1_fashion_train_std, alpha=0.2,
                                 color="darkorange", lw=lw)
                ax4.semilogx(parameter_range, kernel_1_fashion_test_mean, label="Cross-validation score",
                             color="navy", lw=lw)
                ax4.fill_between(parameter_range, kernel_1_fashion_test_mean - kernel_1_fashion_test_std,
                                 kernel_1_fashion_test_mean + kernel_1_fashion_test_std, alpha=0.2,
                                 color="navy", lw=lw)
                ax4.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
                
                ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
                ax2.grid(which='major', linestyle='-', linewidth='0.5', color='white')
                ax3.grid(which='major', linestyle='-', linewidth='0.5', color='white')
                ax4.grid(which='major', linestyle='-', linewidth='0.5', color='white')
                plt.tight_layout()
                # endregion
        
        plt.tight_layout()
        plt.savefig(f"{dataset_directory}/Complexity_Analysis/_{model_name}_{orientation}_{extra_name}_.png", bbox_inches='tight')
        return
    except Exception as e:
        print(f"Exception in plot_combined_complexity:\n", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def plot_combined_confusion_matrix(mnist_clf, mnist_X, mnist_y, fashion_clf, fashion_X, fashion_y,
                                   cmap=None, directory=None, extra_name="", fmt="d", plot_width=8, plot_height=8,
                                   folder="SVM"):
    try:
        from sklearn.metrics import plot_confusion_matrix
        plt.close("all")
        mnist_conf_matrix = plot_confusion_matrix(mnist_clf, mnist_X, mnist_y, cmap=cmap, values_format=fmt)
        plt.savefig(f"{os.getcwd()}/{directory}/Confusion_Matrix/{extra_name}_Fashion_MNIST_Confusion_Matrix.png")
        
        plt.close('all')
        fashion_conf_matrix = plot_confusion_matrix(fashion_clf, fashion_X, fashion_y, cmap=cmap, values_format=fmt)
        plt.savefig(f"{os.getcwd()}/{directory}/Confusion_Matrix/{extra_name}_Fashion_MNIST_Confusion_Matrix.png")
        
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plot_width, plot_height))
        plot_confusion_matrix(mnist_clf, mnist_X, mnist_y, cmap=cmap, ax=ax1, values_format=fmt)
        plot_confusion_matrix(fashion_clf, fashion_X, fashion_y, cmap=cmap, ax=ax2, values_format=fmt)
        ax1.set_title("MNIST \nConfusion Matrix", weight='bold')
        ax2.set_title("Fashion MNIST \nConfusion Matrix", weight='bold')
        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/{directory}/Confusion_Matrix/{extra_name}_Combined_Confusion_Matrix.png")
    except Exception as e:
        print(f"Exception in plot_combined_confusion_matrix:\n", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def plot_precision_recall(classifier, trainX, trainY, testX, testY, training_sizes=np.linspace(0.05, 1.00, 20),
                          folder="SVM", n_classes=10, plot_title=None, dataset_name='MNIST', plot_width=12,
                          plot_height=6, is_final=False, cv=5):
    try:
        idx = [int(trainX.shape[0] * percent) for percent in training_sizes]
        precisions = pd.DataFrame(data=np.zeros(shape=(training_sizes.shape[0], n_classes)),
                                  columns=[f"class {i}" for i in range(n_classes)], index=idx)
        recalls = pd.DataFrame(data=np.zeros(shape=(training_sizes.shape[0], n_classes)),
                               columns=[f"class {i}" for i in range(n_classes)], index=idx)
        print(f"Begin Processing Precision Recall: \n")
        start_time = time.time()
        for i in range(training_sizes.shape[0]):
            print(f"\t\tIteration: {i}\t\tTraining Size: {idx[i]}")
            temp_X = trainX.iloc[:idx[i], :]
            temp_y = trainY.iloc[:idx[i]]
            classifier.fit(temp_X, temp_y)
            
            y_pred = classifier.predict(testX)
            precision = precision_score(y_true=testY, y_pred=y_pred, average=None,
                                        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], zero_division=0)
            precisions.iloc[i] = precision
            
            recall = recall_score(y_true=testY, y_pred=y_pred, average=None,
                                  labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], zero_division=0)
            recalls.iloc[i] = recall

        f1_score = ((precisions * recalls) / (precisions + recalls)) * 2
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"End Processing Precision Recall \n")
        print(f"Elapsed Time: {elapsed_time}")
        
        if plot_title is None:
            plot_title = folder
        plt.close("all")
        if is_final:
            extra = "Final"
        else:
            extra = ""
        
        cm = plt.get_cmap('tab20c')
        colors = [cm(1. * i / 10) for i in range(10)]
        
        f0, ax0 = plt.subplots()
        plt.style.use("ggplot")
        ax0.set_prop_cycle(cycler('color', colors))
        precisions.plot(ax=ax0)
        plt.title(f"{plot_title}\n {dataset_name} Precision", fontsize=15, weight='bold')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        plt.xlabel("Training Size", fontsize=15, weight='heavy')
        plt.ylabel("Precision Score", fontsize=15, weight='heavy')
        plt.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/figures/{folder}/Metrics/_{folder}_Precision_{dataset_name}_{dataset_name}_{extra}.png",
                    bbox_inches='tight')
        
        plt.close("all")
        
        f1, ax1 = plt.subplots()
        plt.style.use("ggplot")
        ax1.set_prop_cycle(cycler('color', colors))
        recalls.plot(ax=ax1)
        plt.title(f"{plot_title}\n {dataset_name} Recall", fontsize=15, weight='bold')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        plt.xlabel("Training Size", fontsize=15, weight='heavy')
        plt.ylabel("Recall Score", fontsize=15, weight='heavy')
        plt.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/figures/{folder}/Metrics/_{folder}_Recall_{dataset_name}_{extra}.png", bbox_inches='tight')
        
        plt.close("all")
        
        f2, ax2 = plt.subplots()
        plt.style.use("ggplot")
        ax2.set_prop_cycle(cycler('color', colors))
        f1_score.plot(ax=ax2)
        plt.title(f"{plot_title}\n {dataset_name} F1 Score", fontsize=15, weight='bold')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        plt.xlabel("Training Size", fontsize=15, weight='heavy')
        plt.ylabel("F1 Score", fontsize=15, weight='heavy')
        plt.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/figures/{folder}/Metrics/_{folder}_F1Score_{dataset_name}_{extra}.png", bbox_inches='tight')
        plt.close("all")
        
        for i in range(1):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(plot_width, plot_height))
            cm = plt.get_cmap('tab20c')
            colors = [cm(1. * i / 10) for i in range(10)]
            
            ax1.set_prop_cycle(cycler('color', colors))
            precisions.plot(ax=ax1)
            ax1.set_title(f"{plot_title}\n {dataset_name} Precision", fontsize=15, weight='bold')
            ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
            ax1.set_xlabel("Training Size", fontsize=15, weight='heavy')
            ax1.set_ylabel("Precision Score", fontsize=15, weight='heavy')
            ax1.set_ylim(0.4, 1.01)
            ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
            
            ax2.set_prop_cycle(cycler('color', colors))
            recalls.plot(ax=ax2)
            ax2.set_title(f"{plot_title}\n {dataset_name} Recall", fontsize=15, weight='bold')
            ax2.grid(which='major', linestyle='-', linewidth='0.5', color='white')
            ax2.set_xlabel("Training Size", fontsize=15, weight='heavy')
            ax2.set_ylabel("Recall Score", fontsize=15, weight='heavy')
            ax2.set_ylim(0.4, 1.01)
            ax2.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
            plt.tight_layout()
            
            ax3.set_prop_cycle(cycler('color', colors))
            f1_score.plot(ax=ax3)
            ax3.set_title(f"{plot_title}\n {dataset_name} F1 Score", fontsize=15, weight='bold')
            ax3.grid(which='major', linestyle='-', linewidth='0.5', color='white')
            ax3.set_xlabel("Training Size", fontsize=15, weight='heavy')
            ax3.set_ylabel("F1 Score", fontsize=15, weight='heavy')
            ax3.set_ylim(0.4, 1.01)
            ax3.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
            plt.tight_layout()
            plt.savefig(f"{os.getcwd()}/figures/{folder}/Metrics/_{folder}_Combined_Precision_Recall_{dataset_name}_{extra}.png",
                        bbox_inches='tight')
        
        return precisions, recalls, f1_score
    
    except Exception as e:
        print(f"Exception in plot_combined_complexity:\n", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
