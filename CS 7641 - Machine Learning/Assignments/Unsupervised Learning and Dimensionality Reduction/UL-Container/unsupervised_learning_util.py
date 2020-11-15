import glob
import os
import pickle
import sys
import time
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
import cycler

from joblib import delayed, Parallel, parallel_backend
from matplotlib import gridspec
from scipy.stats import kurtosis
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_recall_fscore_support, precision_score
from sklearn.metrics import recall_score, silhouette_score, homogeneity_completeness_v_measure, calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split, validation_curve
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from yellowbrick.model_selection import LearningCurve
from yellowbrick.utils.kneed import KneeLocator

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500

# plt.tight_layout()

# From Hands on Machine Learning chapter 3 classification
# https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb
PROJECT_ROOT_DIR = "."
SEED = 42


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


def generate_cv_sets(data_X, data_y, cv_sets, train_limit, validation_pct=0.2):
    try:
        cv_results = {}
        cv_idx = np.random.choice(data_X.shape[0], size=(cv_sets, train_limit), replace=True)
        for i in range(len(cv_idx)):
            temp_results = {"Train_X": None, "Train_Y": None,
                            "Validation_X": None, "Validation_Y": None}
            train_X = data_X.iloc[cv_idx[i]]
            train_y = data_y.iloc[cv_idx[i]]
            train_X.reset_index(drop=True, inplace=True)
            train_y.reset_index(drop=True, inplace=True)
            temp_results["Train_X"] = train_X
            temp_results["Train_Y"] = train_y
            valid_count = int(train_X.shape[0] * validation_pct)
            valid_idx = np.random.choice(train_X.shape[0], valid_count, replace=True)
            valid_X = train_X.iloc[valid_idx]
            valid_y = train_y.iloc[valid_idx]
            valid_X.reset_index(drop=True, inplace=True)
            valid_y.reset_index(drop=True, inplace=True)

            temp_results["Validation_X"] = valid_X
            temp_results["Validation_Y"] = valid_y

            cv_results[f"CV_{i}"] = temp_results

        print("Finished making cross-validation sets")
        print(f"\tEach validation set is a {validation_pct * 100:.4f}% subset of the training set, with replacement")
        return cv_results
    except Exception as generate_cv_sets_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("Exception in 'generate_cv_sets'.\n", generate_cv_sets_exception)
        print()


def split_data(X, y, scale=False, normalize=False, minMax=False, oneHot=False):
    try:
        if scale and not normalize and not minMax and not oneHot:
            X = RobustScaler().fit_transform(X)
        elif normalize and not scale and not minMax and not oneHot:
            X /= max(X.max())
        elif minMax and not scale and not normalize:
            X = MinMaxScaler().fit_transform(X)
            if oneHot:
                y = OneHotEncoder().fit_transform(y.to_numpy().reshape(-1, 1)).todense()
        if oneHot:
            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=SEED)
        else:
            train_X, test_X, train_y, test_y = train_test_split(X, y.to_numpy(), test_size=0.20, random_state=SEED)
        test_X, valid_X = np.split(test_X.astype(np.float32), 2)
        test_y, valid_y = np.split(test_y.astype(np.uint8), 2)
        if oneHot:
            return pd.DataFrame(train_X), pd.DataFrame(train_y), pd.DataFrame(valid_X), \
                   pd.DataFrame(valid_y), pd.DataFrame(test_X), pd.DataFrame(test_y)
        else:
            return pd.DataFrame(train_X), pd.Series(train_y), pd.DataFrame(valid_X), \
                   pd.Series(valid_y), pd.DataFrame(test_X), pd.Series(test_y)
    except Exception as e:
        print(f"Exception in split_data:\n", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def check_folder(_directory):
    MYDIR = os.getcwd() + "\\" + _directory
    CHECK_FOLDER = os.path.isdir(MYDIR)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    else:
        print(MYDIR, "folder already exists.")


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", valfmt="{x:.2f}",
            textcolors=["white", "black"], threshold=None, x_label=None, y_label=None, title=None,
            filename="", folder=None, cmap="viridis", annotate=False, title_size=15, axis_size=15,
            cbar_fontsize=15, set_label_tick_marks=False, **kwargs):
    """
	https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

	Create a heatmap from a numpy array and two lists of labels.

	Parameters
	----------
	data
		A 2D numpy array of shape (N, M).
	row_labels
		A list or array of length N with the labels for the rows.
	col_labels
		A list or array of length M with the labels for the columns.
	ax
		A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
		not provided, use current axes or create a new one.  Optional.
	cbar_kw
		A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
	cbarlabel
		The label for the colorbar.  Optional.
	**kwargs
		All other arguments are forwarded to `imshow`.
	"""
    print("Starting Heatmap")
    if not ax:
        ax = plt.gca()

    plt.style.use("ggplot")
    # Plot the heatmap
    if title is not None:
        ax.set_title(title, fontsize=title_size, weight='bold')

    im = ax.imshow(data, cmap=cmap)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=axis_size, weight='heavy')

    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=axis_size, weight='heavy')

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", weight="heavy", fontsize=cbar_fontsize)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    if set_label_tick_marks:
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth='1', alpha=0.5)
    ax.grid(which='major', linestyle='--', linewidth='0', color='white', alpha=0)
    ax.tick_params(which="minor", bottom=False, left=False)
    if annotate:
        final_heatmap = annotate_heatmap(im=im, valfmt=valfmt, textcolors=textcolors, threshold=threshold)
    # plt.tight_layout()
    print("Heatmap Finished")
    return ax


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
	https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
	A function to annotate a heatmap.

	Parameters
	----------
	im
		The AxesImage to be labeled.
	data
		Data used to annotate.  If None, the image's data is used.  Optional.
	valfmt
		The format of the annotations inside the heatmap.  This should either
		use the string format method, e.g. "$ {x:.2f}", or be a
		`matplotlib.ticker.Formatter`.  Optional.
	textcolors
		A list or array of two color specifications.  The first is used for
		values below a threshold, the second for those above.  Optional.
	threshold
		Value in data units according to which the colors from textcolors are
		applied.  If None (the default) uses the middle of the colormap as
		separation.  Optional.
	**kwargs
		All other arguments are forwarded to each call to `text` used to create
		the text labels.
	"""

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_scatterPlot(results, data_X, data_y, ax, idx, limit=5000, alpha=0.9,
                     label_font_size=18, label_font_weight="heavy", cmap="tab10", markersize=60, n_clusters=10,
                     use_tsne=False):
    try:
        temp_reducer = TSNE()
        reduced_modified_data = temp_reducer.fit_transform(data_X)

        k_cluster = KMeans(n_clusters=n_clusters)
        clustered_data = k_cluster.fit(reduced_modified_data)

        temp_DF = pd.DataFrame(columns=["X", "y"], data=reduced_modified_data)
        temp_DF["Label"] = clustered_data.labels_
        temp_DF.plot.scatter("X", "y", c="Label", cmap=cmap, ax=ax, alpha=alpha, edgecolors='black', s=markersize)

        all_centers = []
        for center in k_cluster.cluster_centers_:
            distance = np.sqrt(((center[0] - temp_DF["X"]) ** 2) + ((center[1] - temp_DF["y"]) ** 2))
            label = temp_DF.loc[np.argmin(distance), "Label"]
            temp_dict = {"Label": label,
                         "cords": center}
            all_centers.append(temp_dict)

        for i in all_centers:
            ax.annotate(str(i["Label"]), (i["cords"][0], i["cords"][1]),
                        fontsize=label_font_size, weight=label_font_weight)
        return
    except Exception as scatterPlot_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'scatterPlot'", scatterPlot_exception)


def calculate_reconstruction_error(true_X, recovered_X):
    try:
        error = (true_X - recovered_X)
        squared_error = error ** 2
        per_pixel_error = np.mean(squared_error, axis=0)
        per_obs_error = np.mean(squared_error, axis=1)
        avg_obs_error = np.mean(per_obs_error)
        return squared_error, per_pixel_error, per_obs_error, avg_obs_error
    except Exception as calculate_reconstruction_error_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'calculate_reconstruction_error'", calculate_reconstruction_error_exception)


def run_pca(data, keep_variance=0.999, dataset_name="MNIST"):
    try:
        results = PCA(whiten=True, svd_solver="full").fit(data)
        component_df = pd.DataFrame(results.components_)
        variance_df = pd.DataFrame(columns=["Variance"], data=results.explained_variance_ratio_)
        variance_df["CumSum"] = variance_df.cumsum()
        temp = variance_df.loc[variance_df["CumSum"] > keep_variance, "CumSum"].index
        number_of_components_to_keep = temp[0]
        pixel_importance = component_df.idxmax(axis=1)
        transformed_data = PCA(n_components=number_of_components_to_keep,
                               whiten=True, svd_solver="full").fit_transform(data)
        component_idx = pixel_importance.iloc[:number_of_components_to_keep]
        with open(f"{os.getcwd()}/DimensionalityReduction/PCA_{dataset_name}_Reduced_Dataset.pkl", "wb") as output_file:
            pickle.dump(transformed_data, output_file)
            output_file.close()
        returned_data = {"Results": results,
                         "Pixel_Importance": pixel_importance,
                         "Number_Of_Components_To_Keep": number_of_components_to_keep,
                         "Component_Idx": component_idx,
                         "Variance_DF": variance_df,
                         "Kept_Variance": keep_variance}

        with open(f"{os.getcwd()}/DimensionalityReduction/PrincipleComponentAnalysis/"
                  f"{dataset_name}_PCA_Results.pkl", "wb") as output_file:
            pickle.dump(returned_data, output_file)
            output_file.close()

        return returned_data
    except Exception as run_pca_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'run_pca'", run_pca_exception)


def run_ica(data, max_component, load_pkl=False, dataset_name="MNIST"):
    try:
        columns = ["Avg_Kurtosis", "Avg_Reconstruction_Error"]
        if load_pkl:
            with open(f"{os.getcwd()}/DimensionalityReduction/"
                      f"IndependentComponentAnalysis/{dataset_name}_ICA_All_Pixel_Error.pkl", "rb") as input_file:
                all_pixel_error = pickle.load(input_file)
                input_file.close()
            with open(f"{os.getcwd()}/DimensionalityReduction/"
                      f"IndependentComponentAnalysis/{dataset_name}_ICA_All_Obs_Error.pkl", "rb") as input_file:
                all_obs_error = pickle.load(input_file)
                input_file.close()
            with open(f"{os.getcwd()}/DimensionalityReduction/"
                      f"IndependentComponentAnalysis/{dataset_name}_ICA_Results.pkl", "rb") as input_file:
                results = pickle.load(input_file)
                input_file.close()
        else:
            results = pd.DataFrame(columns=columns)
            all_obs_error = {}
            all_pixel_error = {}
        all_kurtosis = np.full(shape=(max_component,), fill_value=0, dtype=object)
        all_square_error = {}
        count = 0
        if load_pkl:
            start_num = results.shape[0]
        else:
            start_num = 1
        if start_num < max_component:
            for n in range(start_num, max_component):
                count += 1
                print(f"ICA with {n} Components")
                temp_results = FastICA(n_components=n, whiten=True).fit(data)
                image = np.reshape(temp_results.mean_, (28, 28))
                transformed_X = pd.DataFrame(temp_results.transform(data))
                _kurtosis = kurtosis(transformed_X, axis=1)
                all_kurtosis[n] = _kurtosis
                avg_kurtosis = np.mean(_kurtosis)
                results.loc[n, "Avg_Kurtosis"] = avg_kurtosis
                recovered_X = pd.DataFrame(temp_results.inverse_transform(transformed_X))
                square_error, per_pix_err, \
                per_obs_err, avg_obs_err = calculate_reconstruction_error(true_X=data, recovered_X=recovered_X)
                all_square_error[f"{n}"] = square_error
                all_pixel_error[f"{n}"] = per_pix_err
                all_obs_error[f"{n}"] = per_obs_err
                results.loc[n, "Avg_Reconstruction_Error"] = avg_obs_err
                if count >= 20:
                    count = 0
                    with open(f"{os.getcwd()}/DimensionalityReduction/"
                              f"IndependentComponentAnalysis/{dataset_name}_ICA_All_Pixel_Error.pkl", "wb") as output:
                        pickle.dump(all_pixel_error, output)
                        output.close()
                    with open(f"{os.getcwd()}/DimensionalityReduction/"
                              f"IndependentComponentAnalysis/{dataset_name}_ICA_All_Obs_Error.pkl", "wb") as output:
                        pickle.dump(all_obs_error, output)
                        output.close()
                    with open(f"{os.getcwd()}/DimensionalityReduction/"
                              f"IndependentComponentAnalysis/{dataset_name}_ICA_Results.pkl", "wb") as output:
                        pickle.dump(results, output)
                        output.close()

        values = {"Results": results,
                  "All_Obs_Error": all_obs_error,
                  "All_Pixel_Error": all_pixel_error}
        return values
    except Exception as run_ica_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'run_ica'", run_ica_exception)


def run_randomized_projections(data, max_components=100, num_retrys=10, dataset_name="MNIST", load_pkl=False):
    try:
        count = 0
        if load_pkl:
            with open(f"{os.getcwd()}/DimensionalityReduction/"
                      f"RandomProjections/{dataset_name}_RP_results.pkl", "rb") as input_file:
                error_df = pickle.load(input_file)
                input_file.close()
            start_val = error_df.shape[0]
        else:
            columns = ["Avg_Error"]
            error_df = pd.DataFrame(columns=columns)
            start_val = 1
        for n in range(start_val, max_components + 1):
            print(f"Current Components: {n}  /  {max_components}")
            temp_error = np.zeros(shape=(num_retrys,))
            count += 1
            for retry in range(num_retrys):
                random_proj = GaussianRandomProjection(n_components=n).fit(data)
                transformed_data = pd.DataFrame(random_proj.transform(data))
                inverse = pd.DataFrame(np.linalg.pinv(random_proj.components_.T))
                reconstructed_data = transformed_data.dot(inverse)
                error = mean_squared_error(data, reconstructed_data)
                temp_error[retry] = error
            avg_err = np.mean(temp_error)
            print(f"\tAverage Error: {avg_err:.3f}")
            error_df.loc[n, "Avg_Error"] = avg_err
            if count >= 20:
                error_df.plot()
                plt.show()
                count = 0
                with open(f"{os.getcwd()}/DimensionalityReduction/"
                          f"RandomProjections/{dataset_name}_RP_results.pkl", "wb") as output_file:
                    pickle.dump(error_df, output_file)
                    output_file.close()
        with open(f"{os.getcwd()}/DimensionalityReduction/"
                  f"RandomProjections/{dataset_name}_RP_results.pkl", "wb") as output_file:
            pickle.dump(error_df, output_file)
            output_file.close()
        return
    except Exception as run_randomized_projections_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'run_randomized_projections'", run_randomized_projections_exception)


def run_random_forest(data_X, data_y, valid_X, valid_y, max_components=100, dataset_name="MNIST", load_pkl=False):
    try:
        RF_Classifier = RandomForestClassifier(n_estimators=1000, n_jobs=-1, verbose=10, warm_start=True)

        RF_Classifier.fit(data_X, data_y)

        Random_Forest_DF = pd.DataFrame(columns=["Feature_Importance"],
                                        data=RF_Classifier.feature_importances_)

        Random_Forest_DF.sort_values(by=["Feature_Importance"], ascending=False, inplace=True)
        random_forest_index = Random_Forest_DF.copy().index
        Random_Forest_DF["Feature_Importance_Index"] = random_forest_index

        Random_Forest_DF.reset_index(drop=True, inplace=True)
        all_idx = []
        dt_count = 0
        svm_count = 0
        knn_count = 0
        count = 0
        for i in range(Random_Forest_DF.shape[0]):
            count += 1
            print(f"Current index: {i}")
            all_idx.append(random_forest_index[i])
            temp_X = valid_X.iloc[:, all_idx]
            temp_y = valid_y.iloc[:]
            if dt_count < 4:
                decision_tree = DecisionTreeClassifier().fit(temp_X, temp_y)
                dt_score = decision_tree.score(temp_X, temp_y)
                if dt_score >= 1.0:
                    dt_count += 1
                Random_Forest_DF.loc[i, "Decision_Tree_Accuracy"] = dt_score
            if knn_count < 4:
                knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1).fit(temp_X, temp_y)
                knn_score = knn.score(temp_X, temp_y)
                if knn_score >= 1.0:
                    knn_count += 1
                Random_Forest_DF.loc[i, "KNN_Accuracy"] = knn_score
            if svm_count < 4:
                svm = SVC().fit(temp_X, temp_y)
                svm_score = svm.score(temp_X, temp_y)
                if svm_score >= 1.0:
                    svm_score += 1
                Random_Forest_DF.loc[i, "SVM_Accuracy"] = svm_score

            if count >= 20:
                count = 0
                with open(f"{os.getcwd()}/DimensionalityReduction/RandomForest/RandomForest_{dataset_name}_Results.pkl",
                          "wb") as output_file:
                    pickle.dump(Random_Forest_DF, output_file)
                    output_file.close()
        Random_Forest_DF.fillna(method="ffill", inplace=True)
        with open(f"{os.getcwd()}/DimensionalityReduction/RandomForest/RandomForest_{dataset_name}_Results.pkl",
                  "wb") as output_file:
            pickle.dump(Random_Forest_DF, output_file)
            output_file.close()
        print()
        return
    except Exception as run_random_forest_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'run_random_forest'", run_random_forest_exception)


def plot_pca_results(mnist_results, fashion_results, mnist_X, mnist_y, fashion_X, fashion_y):
    try:
        # region Part 1 Combined
        plt.close("all")
        fig, ((ax1_1, ax1_2), (ax2_1, ax2_2)) = plt.subplots(2, 2, figsize=(12, 12))
        ax1_1.plot(mnist_results["Variance_DF"]["CumSum"], label="Explained Variance")

        image = np.reshape(mnist_results["Results"].mean_, (28, 28))
        heatmap(image,
                row_labels=np.arange(0, 28),
                col_labels=np.arange(0, 28),
                ax=ax1_2, cbarlabel="Importance",
                title=f"Pixel Importance\nMNIST",
                folder="DimensionalityReduction/PrincipleComponentAnalysis",
                filename=f"Mnist_Pixel_Importance",
                cmap="inferno")
        ax1_1.axhline(mnist_results["Kept_Variance"], color="black", linestyle="--", alpha=0.5, lw=2,
                      label=f"Explained Variance @ {mnist_results['Kept_Variance']:.3f}")

        ax1_1.axvline(mnist_results["Number_Of_Components_To_Keep"], color="navy", linestyle="--", alpha=0.5, lw=2,
                      label=f"Number of Components @ {mnist_results['Number_Of_Components_To_Keep']}")

        ax1_1.set_title(f"Explained Variance Vs. Number of Components\nMNIST",
                        fontsize=15, weight='bold')
        ax1_1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax1_1.set_xlabel(f"N Components", fontsize=15, weight='heavy')
        ax1_1.set_ylabel(f"Explained Variance", fontsize=15, weight='heavy')
        ax1_1.legend(loc="best", markerscale=1.1, frameon=True,
                     edgecolor="black", fancybox=True, shadow=True)

        ax2_1.plot(fashion_results["Variance_DF"]["CumSum"], label="Explained Variance")
        image = np.reshape(fashion_results["Results"].mean_, (28, 28))
        heatmap(image,
                row_labels=np.arange(0, 28),
                col_labels=np.arange(0, 28),
                ax=ax2_2, cbarlabel="Importance",
                title=f"Pixel Importance\nFashion MNIST",
                folder="DimensionalityReduction/PrincipleComponentAnalysis",
                filename=f"Fashion_Mnist_Pixel_Importance",
                cmap="inferno")
        ax2_1.axhline(fashion_results["Kept_Variance"], color="black", linestyle="--", alpha=0.5, lw=2,
                      label=f"Explained Variance @ {fashion_results['Kept_Variance']:.3f}")
        ax2_1.axvline(fashion_results["Number_Of_Components_To_Keep"], color="navy",
                      linestyle="--", alpha=0.5, lw=2,
                      label=f"Number of Components @ {fashion_results['Number_Of_Components_To_Keep']}")

        ax2_1.set_title(f"Explained Variance Vs. Number of Components\nFashion MNIST",
                        fontsize=15, weight='bold')
        ax2_1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax2_1.set_xlabel(f"N Components", fontsize=15, weight='heavy')
        ax2_1.set_ylabel(f"Explained Variance", fontsize=15, weight='heavy')
        ax2_1.legend(loc="best", markerscale=1.1, frameon=True,
                     edgecolor="black", fancybox=True, shadow=True)
        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/DimensionalityReduction/"
                    f"PrincipleComponentAnalysis/PCA_Results_Combined.png")
        # endregion

        # region Part 2 Separate
        # region MNIST
        plt.close("all")
        fig, ((ax1_1, ax1_2), (ax2_1, ax2_2)) = plt.subplots(2, 2, figsize=(12, 12))
        ax1_1.plot(mnist_results["Variance_DF"]["CumSum"], label="Explained Variance")

        image = np.reshape(mnist_results["Results"].mean_, (28, 28))
        heatmap(image,
                row_labels=np.arange(0, 28),
                col_labels=np.arange(0, 28),
                ax=ax1_2, cbarlabel="Importance",
                title=f"Pixel Importance\nMNIST",
                folder="DimensionalityReduction/PrincipleComponentAnalysis",
                filename=f"Mnist_Pixel_Importance",
                cmap="inferno")
        ax1_1.axhline(mnist_results["Kept_Variance"], color="black", linestyle="--", alpha=0.5, lw=2,
                      label=f"Explained Variance @ {mnist_results['Kept_Variance']:.3f}")

        ax1_1.axvline(mnist_results["Number_Of_Components_To_Keep"], color="navy", linestyle="--", alpha=0.5, lw=2,
                      label=f"Number of Components @ {mnist_results['Number_Of_Components_To_Keep']}")

        ax1_1.set_title(f"Explained Variance Vs. Number of Components\nMNIST",
                        fontsize=15, weight='bold')
        ax1_1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax1_1.set_xlabel(f"N Components", fontsize=15, weight='heavy')
        ax1_1.set_ylabel(f"Explained Variance", fontsize=15, weight='heavy')
        ax1_1.legend(loc="best", markerscale=1.1, frameon=True,
                     edgecolor="black", fancybox=True, shadow=True)

        rand_idx = np.random.randint(0, 784, 1)
        pred_X = mnist_results["Results"].transform(mnist_X.iloc[rand_idx])
        recon_X = mnist_results["Results"].inverse_transform(pred_X)
        ax2_2.imshow(np.reshape(recon_X, (28, 28)), cmap="gray")
        ax2_1.imshow(np.reshape(mnist_X.iloc[rand_idx].to_numpy(), (28, 28)), cmap="gray")
        ax2_1.set_title(f"Training Set\nMNIST ",
                        fontsize=15, weight='bold')
        ax2_2.set_title(f"Feature Spaced Reduced by "
                        f"{784 - mnist_results['Number_Of_Components_To_Keep']}\n"
                        f"MNIST",
                        fontsize=15, weight='bold')
        ax2_1.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)
        ax2_2.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)
        ax2_1.set_xlabel(f"Label={mnist_y.loc[rand_idx].values[0]}", fontsize=15, weight='heavy')
        ax2_2.set_xlabel(f"Label={mnist_y.loc[rand_idx].values[0]}", fontsize=15, weight='heavy')
        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/DimensionalityReduction/"
                    f"PrincipleComponentAnalysis/PCA_MNIST_Recon_Compare.png")
        # endregion

        # region Fashion
        plt.close("all")
        fig, ((ax1_1, ax1_2), (ax2_1, ax2_2)) = plt.subplots(2, 2, figsize=(12, 12))
        ax1_1.plot(fashion_results["Variance_DF"]["CumSum"], label="Explained Variance")

        image = np.reshape(fashion_results["Results"].mean_, (28, 28))
        heatmap(image,
                row_labels=np.arange(0, 28),
                col_labels=np.arange(0, 28),
                ax=ax1_2, cbarlabel="Importance",
                title=f"Pixel Importance\nFashion MNIST",
                folder="DimensionalityReduction/PrincipleComponentAnalysis",
                filename=f"Fashion_Pixel_Importance",
                cmap="inferno")
        ax1_1.axhline(fashion_results["Kept_Variance"], color="black", linestyle="--", alpha=0.5, lw=2,
                      label=f"Explained Variance @ {fashion_results['Kept_Variance']:.3f}")

        ax1_1.axvline(fashion_results["Number_Of_Components_To_Keep"], color="navy", linestyle="--", alpha=0.5, lw=2,
                      label=f"Number of Components @ {fashion_results['Number_Of_Components_To_Keep']}")

        ax1_1.set_title(f"Explained Variance Vs. Number of Components\nFashion MNIST",
                        fontsize=15, weight='bold')
        ax1_1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax1_1.set_xlabel(f"N Components", fontsize=15, weight='heavy')
        ax1_1.set_ylabel(f"Explained Variance", fontsize=15, weight='heavy')
        ax1_1.legend(loc="best", markerscale=1.1, frameon=True,
                     edgecolor="black", fancybox=True, shadow=True)

        fashion_labels = {0: "T-shirt/top",
                          1: "Trouser",
                          2: "Pullover",
                          3: "Dress",
                          4: "Coat",
                          5: "Sandal",
                          6: "Shirt",
                          7: "Sneaker",
                          8: "Bag",
                          9: "Ankle Boot"}

        rand_idx = np.random.randint(0, 784, 1)
        pred_X = fashion_results["Results"].transform(fashion_X.iloc[rand_idx])
        recon_X = fashion_results["Results"].inverse_transform(pred_X)
        ax2_2.imshow(np.reshape(recon_X, (28, 28)), cmap="gray")
        ax2_1.imshow(np.reshape(fashion_X.iloc[rand_idx].to_numpy(), (28, 28)), cmap="gray")
        ax2_1.set_title(f"Training Set\nMNIST  label={fashion_labels[fashion_y.loc[rand_idx].values[0]]}",
                        fontsize=15, weight='bold')
        ax2_2.set_title(f"Feature Spaced Reduced by "
                        f"{784 - fashion_results['Number_Of_Components_To_Keep']}\n"
                        f"Fashion MNIST",
                        fontsize=15, weight='bold')
        ax2_1.set_xlabel(f"Label={fashion_labels[fashion_y.loc[rand_idx].values[0]]}", fontsize=15, weight='heavy')
        ax2_2.set_xlabel(f"Label={fashion_labels[fashion_y.loc[rand_idx].values[0]]}", fontsize=15, weight='heavy')
        ax2_1.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)
        ax2_2.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/DimensionalityReduction/"
                    f"PrincipleComponentAnalysis/PCA_Fashion_Recon_Compare.png")
        # endregion

        # endregion
        return
    except Exception as plot_ica_results_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'plot_ica_results'", plot_ica_results_exception)


def plot_ica_results(mnist_df, fashion_df, mnist_data, fashion_data, mnist_X, mnist_y, fashion_X, fashion_y,
                     mnist_X_scaled, mnist_y_scaled, fashion_X_scaled, fashion_y_scaled, err_lim=0.005):
    try:
        fashion_labels = {0: "T-shirt/top",
                          1: "Trouser",
                          2: "Pullover",
                          3: "Dress",
                          4: "Coat",
                          5: "Sandal",
                          6: "Shirt",
                          7: "Sneaker",
                          8: "Bag",
                          9: "Ankle Boot"}
        # region Combined
        plt.close("all")
        fig, ((ax1_1, ax1_2), (ax2_1, ax2_2)) = plt.subplots(2, 2, figsize=(12, 12))
        ln1 = ax1_1.plot(mnist_df["Avg_Kurtosis"], label="Average Kurtosis")

        ax1_1_secondary = ax1_1.twinx()
        ln2 = ax1_1_secondary.plot(mnist_df["Avg_Reconstruction_Error"],
                                   label="Average Reconstruction Error", color="navy")
        mnist_components = mnist_df.loc[mnist_df["Avg_Reconstruction_Error"] < err_lim,
                                        "Avg_Reconstruction_Error"].index[0]
        # Getting image b/c I forgot to save data
        temp_results_mnist = FastICA(n_components=mnist_components, whiten=True, max_iter=400).fit(mnist_X_scaled)
        transformed_mnist_dataset = temp_results_mnist.transform(mnist_X_scaled)
        with open(f"{os.getcwd()}/DimensionalityReduction/ICA_MNIST_Reduced_Dataset.pkl", "wb") as output_file:
            pickle.dump(transformed_mnist_dataset, output_file)
            output_file.close()
        image = np.reshape(temp_results_mnist.mean_, (28, 28))
        heatmap(image,
                row_labels=np.arange(0, 28),
                col_labels=np.arange(0, 28),
                ax=ax1_2, cbarlabel="Importance",
                title=f"MNIST\nPixel Importance",
                folder="DimensionalityReduction/IndependentComponentAnalysis",
                filename=f"Mnist_Pixel_Importance",
                kwargs={"cmap": "inferno"},
                cmap="inferno")
        ln3 = ax1_1.axvline(mnist_components, color="black", linestyle="--", alpha=0.5, lw=2,
                            label=f"Max_Kurtosis @ {mnist_components:.2f}")
        ax1_1_secondary.annotate("Error @ 1%",
                                 (mnist_components + 1, mnist_df.loc[mnist_components, "Avg_Reconstruction_Error"]),
                                 fontsize=8, weight='heavy')

        ax1_1.set_title(f"Kurtosis Vs. Reconstruction Error\nMNIST",
                        fontsize=15, weight='bold')
        ax1_1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax1_1_secondary.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.05)
        ax1_1.set_xlabel(f"N Components", fontsize=15, weight='heavy')
        ax1_1.set_ylabel(f"Kurtosis/Reconstruction Error", fontsize=15, weight='heavy')
        lns = ln1 + ln2
        lns.append(ln3)
        labs = [l.get_label() for l in lns]
        ax1_1.legend(lns, labs, loc="best", markerscale=1.1, frameon=True,
                     edgecolor="black", fancybox=True, shadow=True)

        ln1 = ax2_1.plot(fashion_df["Avg_Kurtosis"], label="Average Kurtosis")
        ax2_1_secondary = ax2_1.twinx()
        ln2 = ax2_1_secondary.plot(fashion_df["Avg_Reconstruction_Error"],
                                   label="Average Reconstruction Error", color="navy")
        fashion_components = fashion_df.loc[fashion_df["Avg_Reconstruction_Error"] < err_lim,
                                            "Avg_Reconstruction_Error"].index[0]
        # Getting image b/c I forgot to save data
        temp_results_fashion = FastICA(n_components=fashion_components, whiten=True, max_iter=400).fit(fashion_X_scaled)
        transformed_fashion_dataset = temp_results_fashion.transform(fashion_X_scaled)
        with open(f"{os.getcwd()}/DimensionalityReduction/ICA_Fashion_Reduced_Dataset.pkl", "wb") as output_file:
            pickle.dump(transformed_fashion_dataset, output_file)
            output_file.close()
        image = np.reshape(temp_results_fashion.mean_, (28, 28))
        heatmap(image,
                row_labels=np.arange(0, 28),
                col_labels=np.arange(0, 28),
                ax=ax2_2, cbarlabel="Importance",
                title=f"Fashion MNIST\nPixel Importance",
                folder="DimensionalityReduction/IndependentComponentAnalysis",
                filename=f"Fashion_Mnist_Pixel_Importance",
                kwargs={"cmap": "inferno"},
                cmap="inferno")
        ln3 = ax2_1.axvline(fashion_components, color="black", linestyle="--", alpha=0.5, lw=2,
                            label=f"Max_Kurtosis @ {fashion_components:.2f}")
        ax2_1_secondary.annotate("Error @ 1%",
                                 (fashion_components + 1,
                                  fashion_df.loc[fashion_components, "Avg_Reconstruction_Error"]),
                                 fontsize=8, weight='heavy')

        ax2_1.set_title(f"Kurtosis Vs. Reconstruction Error\nFashion MNIST",
                        fontsize=15, weight='bold')
        ax2_1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax2_1_secondary.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.05)
        ax2_1.set_xlabel(f"N Components", fontsize=15, weight='heavy')
        ax2_1.set_ylabel(f"Kurtosis/Reconstruction Error", fontsize=15, weight='heavy')
        lns = ln1 + ln2
        lns.append(ln3)
        labs = [l.get_label() for l in lns]
        ax2_1.legend(lns, labs, loc="best", markerscale=1.1, frameon=True,
                     edgecolor="black", fancybox=True, shadow=True)

        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/DimensionalityReduction/"
                    f"IndependentComponentAnalysis/ICA_Results_Kurtosis_ReconError_final.png")
        # endregion

        # region Separate

        # region MNIST
        plt.close("all")
        dataset_name = "MNIST"
        fig, ((ax1_1, ax1_2), (ax2_1, ax2_2)) = plt.subplots(2, 2, figsize=(12, 12))
        ln1 = ax1_1.plot(mnist_df["Avg_Kurtosis"], label="Average Kurtosis")

        ax1_1_secondary = ax1_1.twinx()
        ln2 = ax1_1_secondary.plot(mnist_df["Avg_Reconstruction_Error"],
                                   label="Average Reconstruction Error", color="navy")
        # Getting image b/c I forgot to save data

        image = np.reshape(temp_results_mnist.mean_, (28, 28))
        heatmap(image,
                row_labels=np.arange(0, 28),
                col_labels=np.arange(0, 28),
                ax=ax1_2, cbarlabel="Importance",
                title=f"Pixel Importance\n{dataset_name}",
                folder="DimensionalityReduction/IndependentComponentAnalysis",
                filename=f"Mnist_Pixel_Importance",
                kwargs={"cmap": "inferno"},
                cmap="inferno")
        ln3 = ax1_1.axvline(mnist_components, color="black", linestyle="--", alpha=0.5, lw=2,
                            label=f"Max_Kurtosis @ {mnist_components:.2f}")
        ax1_1_secondary.annotate("Error @ 1%",
                                 (mnist_components + 1, mnist_df.loc[mnist_components, "Avg_Reconstruction_Error"]),
                                 fontsize=8, weight='heavy')
        ax1_1.set_title(f"Kurtosis Vs. Reconstruction Error\nMNIST",
                        fontsize=15, weight='bold')
        ax1_1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax1_1_secondary.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.05)
        ax1_1.set_xlabel(f"N Components", fontsize=15, weight='heavy')
        ax1_1.set_ylabel(f"Kurtosis/Reconstruction Error", fontsize=15, weight='heavy')
        lns = ln1 + ln2
        lns.append(ln3)
        labs = [l.get_label() for l in lns]
        ax1_1.legend(lns, labs, loc="best", markerscale=1.1, frameon=True,
                     edgecolor="black", fancybox=True, shadow=True)

        rand_idx = np.random.randint(0, 784, 1)
        ax2_1.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)
        ax2_2.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)
        recon_X = temp_results_mnist.inverse_transform(transformed_mnist_dataset)[rand_idx]

        ax2_2.imshow(np.reshape(recon_X, (28, 28)), cmap="gray")
        ax2_1.imshow(np.reshape(mnist_X.iloc[rand_idx].to_numpy(), (28, 28)), cmap="gray")

        ax2_1.set_title(f"Training Set\nMNIST",
                        fontsize=15, weight='bold')
        ax2_2.set_title(f"Feature Spaced Reduced by "
                        f"{784 - mnist_components}\n"
                        f"MNIST",
                        fontsize=15, weight='bold')
        ax2_1.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)
        ax2_2.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)
        ax2_1.set_xlabel(f"Label={mnist_y.loc[rand_idx].values[0]}", fontsize=15, weight='heavy')
        ax2_2.set_xlabel(f"Label={mnist_y.loc[rand_idx].values[0]}", fontsize=15, weight='heavy')

        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/DimensionalityReduction/"
                    f"IndependentComponentAnalysis/ICA_{dataset_name}_ReconError_final.png")
        # endregion
        # region Fashion
        plt.close("all")
        dataset_name = "Fashion-MNIST"
        fig, ((ax1_1, ax1_2), (ax2_1, ax2_2)) = plt.subplots(2, 2, figsize=(12, 12))
        ln1 = ax1_1.plot(fashion_df["Avg_Kurtosis"], label="Average Kurtosis")

        ax1_1_secondary = ax1_1.twinx()
        ln2 = ax1_1_secondary.plot(fashion_df["Avg_Reconstruction_Error"],
                                   label="Average Reconstruction Error", color="navy")
        # Getting image b/c I forgot to save data
        image = np.reshape(temp_results_fashion.mean_, (28, 28))
        heatmap(image,
                row_labels=np.arange(0, 28),
                col_labels=np.arange(0, 28),
                ax=ax1_2, cbarlabel="Importance",
                title=f"Pixel Importance\n{dataset_name}",
                folder="DimensionalityReduction/IndependentComponentAnalysis",
                filename=f"{dataset_name}_Pixel_Importance",
                kwargs={"cmap": "inferno"},
                cmap="inferno")
        ln3 = ax1_1.axvline(fashion_components, color="black", linestyle="--", alpha=0.5, lw=2,
                            label=f"Max_Kurtosis @ {fashion_components:.2f}")

        ax1_1_secondary.annotate("Error @ 1%",
                                 (fashion_components + 1,
                                  fashion_df.loc[fashion_components, "Avg_Reconstruction_Error"]),
                                 fontsize=8, weight='heavy')
        ax1_1.set_title(f"Kurtosis Vs. Reconstruction Error\n{dataset_name}",
                        fontsize=15, weight='bold')
        ax1_1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax1_1_secondary.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.05)
        ax1_1.set_xlabel(f"N Components", fontsize=15, weight='heavy')
        ax1_1.set_ylabel(f"Kurtosis/Reconstruction Error", fontsize=15, weight='heavy')
        lns = ln1 + ln2
        lns.append(ln3)
        labs = [l.get_label() for l in lns]
        ax1_1.legend(lns, labs, loc="best", markerscale=1.1, frameon=True,
                     edgecolor="black", fancybox=True, shadow=True)

        rand_idx = np.random.randint(0, 784, 1)
        ax2_1.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)
        ax2_2.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)
        recon_X = temp_results_fashion.inverse_transform(transformed_fashion_dataset)[rand_idx]

        ax2_2.imshow(np.reshape(recon_X, (28, 28)), cmap="gray")
        ax2_1.imshow(np.reshape(fashion_X.iloc[rand_idx].to_numpy(), (28, 28)), cmap="gray")

        ax2_1.set_title(f"Training Set\n{dataset_name}",
                        fontsize=15, weight='bold')
        ax2_2.set_title(f"Feature Spaced Reduced by "
                        f"{784 - fashion_components}\n"
                        f"{dataset_name}",
                        fontsize=15, weight='bold')
        ax2_1.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)
        ax2_2.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)
        ax2_1.set_xlabel(f"Label={fashion_labels[fashion_y.loc[rand_idx].values[0]]}",
                         fontsize=15, weight='heavy')
        ax2_2.set_xlabel(f"Label={fashion_labels[fashion_y.loc[rand_idx].values[0]]}",
                         fontsize=15, weight='heavy')

        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/DimensionalityReduction/"
                    f"IndependentComponentAnalysis/ICA_{dataset_name}_ReconError_final.png")
        # endregion
        # endregion

        return
    except Exception as plot_ica_results_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'plot_ica_results'", plot_ica_results_exception)


def plot_randomized_projection_results(mnist_results, fashion_results, mnist_X, mnist_y, fashion_X, fashion_y):
    try:
        fashion_labels = {0: "T-shirt/top",
                          1: "Trouser",
                          2: "Pullover",
                          3: "Dress",
                          4: "Coat",
                          5: "Sandal",
                          6: "Shirt",
                          7: "Sneaker",
                          8: "Bag",
                          9: "Ankle Boot"}
        # find max error
        mnist_limit_for_ten_percent_error = 0.01
        mnist_component = mnist_results["Avg_Error"][mnist_results["Avg_Error"]
                                                     <= mnist_limit_for_ten_percent_error].index[0]

        random_proj = GaussianRandomProjection(n_components=mnist_component).fit(mnist_X)
        transformed_data = pd.DataFrame(random_proj.transform(mnist_X))
        with open(f"{os.getcwd()}/DimensionalityReduction/RP_MNIST_Reduced_Dataset.pkl", "wb") as output_file:
            pickle.dump(transformed_data, output_file)
            output_file.close()
        inverse = pd.DataFrame(np.linalg.pinv(random_proj.components_.T))
        mnist_reconstructed_data = transformed_data.dot(inverse)
        rand_idx = np.random.randint(0, 784)
        mnist_reconstructed_label = mnist_y.iloc[rand_idx]
        mnist_original = np.reshape(mnist_X.iloc[rand_idx, :].to_numpy(), (28, 28))
        mnist_reconstructed_img = np.reshape(mnist_reconstructed_data.iloc[rand_idx, :].to_numpy(), (28, 28))

        fashion_limit_for_ten_percent_error = 0.01
        fashion_component = fashion_results["Avg_Error"][fashion_results["Avg_Error"]
                                                         <= fashion_limit_for_ten_percent_error].index[0]

        random_proj = GaussianRandomProjection(n_components=fashion_component).fit(fashion_X)
        transformed_data = pd.DataFrame(random_proj.transform(fashion_X))
        with open(f"{os.getcwd()}/DimensionalityReduction/RP_Fashion_Reduced_Dataset.pkl", "wb") as output_file:
            pickle.dump(transformed_data, output_file)
            output_file.close()
        inverse = pd.DataFrame(np.linalg.pinv(random_proj.components_.T))
        fashion_reconstructed_data = transformed_data.dot(inverse)
        rand_idx = np.random.randint(0, 784)
        fashion_reconstructed_label = fashion_labels[fashion_y.iloc[rand_idx]]
        fashion_original = np.reshape(fashion_X.iloc[rand_idx, :].to_numpy(), (28, 28))
        fashion_reconstructed_img = np.reshape(fashion_reconstructed_data.iloc[rand_idx, :].to_numpy(), (28, 28))

        # find threshold of less than 10% recon error
        # region Combined
        plt.close("all")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.plot(mnist_results, label="Avg Reconstruction Error")
        ax1.set_title(f"Reconstruction Error\nMNIST",
                      fontsize=15, weight='bold')
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax1.set_xlabel(f"N Components", fontsize=15, weight='heavy')
        ax1.set_ylabel(f"Reconstruction Error", fontsize=15, weight='heavy')
        ax1.axvline(mnist_component, color="navy", linestyle="--", alpha=0.5, lw=2,
                    label=f"Number of Components @ {mnist_component}")
        ax1.axhline(mnist_limit_for_ten_percent_error, color="black", linestyle="--", alpha=0.5, lw=2,
                    label=f"Threshold of 10% Reconstruction Error")
        ax1.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)

        ax2.plot(fashion_results, label="Avg Reconstruction Error")
        ax2.set_title(f"Reconstruction Error\nFashion MNIST",
                      fontsize=15, weight='bold')
        ax2.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax2.set_xlabel(f"N Components", fontsize=15, weight='heavy')
        ax2.set_ylabel(f"Reconstruction Error", fontsize=15, weight='heavy')
        ax2.axvline(fashion_component, color="navy", linestyle="--", alpha=0.5, lw=2,
                    label=f"Number of Components @ {fashion_component}")
        ax2.axhline(fashion_limit_for_ten_percent_error, color="black", linestyle="--", alpha=0.5, lw=2,
                    label=f"Threshold of 10% Reconstruction Error")
        ax2.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)

        plt.savefig(f"{os.getcwd()}/DimensionalityReduction/RandomProjections/RP_Reconstruction_Error_Combined.png")
        # endregion

        # region Separate
        # region MNIST
        plt.close("all")
        dataset_name = "MNIST"
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[8, 5], figure=fig, wspace=0.05, hspace=0.15)
        ax0 = fig.add_subplot(gs[:, 0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[1, 1])

        ax0.plot(mnist_results, label="Avg Reconstruction Error")
        ax0.set_title(f"Reconstruction Error\n{dataset_name}",
                      fontsize=15, weight='bold')
        ax0.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax0.set_xlabel(f"N Components", fontsize=15, weight='heavy')
        ax0.set_ylabel(f"Reconstruction Error", fontsize=15, weight='heavy')
        ax0.axvline(mnist_component, color="navy", linestyle="--", alpha=0.5, lw=2,
                    label=f"Number of Components @ {mnist_component}")
        ax0.axhline(mnist_limit_for_ten_percent_error, color="black", linestyle="--", alpha=0.5, lw=2,
                    label=f"Threshold of 10% Reconstruction Error")
        ax0.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)
        ax1.imshow(mnist_original, cmap="gray")
        ax1.set_ylabel(f"Label={mnist_reconstructed_label}", fontsize=15, weight='heavy')
        ax1.xaxis.set_major_formatter(plt.NullFormatter())
        ax1.yaxis.set_major_formatter(plt.NullFormatter())
        ax1.yaxis.set_label_position("right")
        ax1.set_title(f"Original Data",
                      fontsize=15, weight='bold')
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)

        ax2.imshow(mnist_reconstructed_img, cmap="gray")
        ax2.set_ylabel(f"Label={mnist_reconstructed_label}", fontsize=15, weight='heavy')
        ax2.xaxis.set_major_formatter(plt.NullFormatter())
        ax2.yaxis.set_major_formatter(plt.NullFormatter())
        ax2.yaxis.set_label_position("right")
        ax2.set_title(f"Reconstructed Data",
                      fontsize=15, weight='bold')
        ax2.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)

        plt.savefig(f"{os.getcwd()}/DimensionalityReduction/RandomProjections/RP_{dataset_name}_Error.png")
        # endregion

        # region Fashion
        plt.close("all")
        dataset_name = "Fashion-MNIST"
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[8, 5], figure=fig, wspace=0.05, hspace=0.15)
        ax0 = fig.add_subplot(gs[:, 0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[1, 1])

        ax0.plot(fashion_results, label="Avg Reconstruction Error")
        ax0.set_title(f"Reconstruction Error\n{dataset_name}",
                      fontsize=15, weight='bold')
        ax0.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax0.set_xlabel(f"N Components", fontsize=15, weight='heavy')
        ax0.set_ylabel(f"Reconstruction Error", fontsize=15, weight='heavy')
        ax0.axvline(fashion_component, color="navy", linestyle="--", alpha=0.5, lw=2,
                    label=f"N Components @ {fashion_component}")
        ax0.axhline(fashion_limit_for_ten_percent_error, color="black", linestyle="--", alpha=0.5, lw=2,
                    label=f"Threshold of 10% Reconstruction Error")
        ax0.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True)
        ax1.imshow(fashion_original, cmap="gray")
        ax1.set_ylabel(f"Label={fashion_reconstructed_label}", fontsize=15, weight='heavy')
        ax1.xaxis.set_major_formatter(plt.NullFormatter())
        ax1.yaxis.set_major_formatter(plt.NullFormatter())
        ax1.yaxis.set_label_position("right")
        ax1.set_title(f"Original Data",
                      fontsize=15, weight='bold')
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)

        ax2.imshow(fashion_reconstructed_img, cmap="gray")
        ax2.set_ylabel(f"Label={fashion_reconstructed_label}", fontsize=15, weight='heavy')
        ax2.xaxis.set_major_formatter(plt.NullFormatter())
        ax2.yaxis.set_major_formatter(plt.NullFormatter())
        ax2.yaxis.set_label_position("right")
        ax2.set_title(f"Reconstructed Data",
                      fontsize=15, weight='bold')
        ax2.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)

        plt.savefig(f"{os.getcwd()}/DimensionalityReduction/RandomProjections/RP_{dataset_name}_Error.png")
        # endregion
        # endregion

        return
    except Exception as plot_randomized_projection_results_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'plot_randomized_projection_results'", plot_randomized_projection_results_exception)


def plot_random_forest_results(mnist_results, fashion_results, mnist_X, mnist_y, fashion_X, fashion_y):
    try:
        mnist_results["KNN_ROC"] = mnist_results["KNN_Accuracy"].pct_change(periods=10)
        mnist_results["SVM_ROC"] = mnist_results["SVM_Accuracy"].pct_change(periods=10)
        mnist_results["DIFF"] = np.abs(mnist_results["KNN_ROC"] - mnist_results["SVM_ROC"])

        temp_mnist = mnist_results[["Decision_Tree_Accuracy", "KNN_Accuracy", "SVM_Accuracy"]].copy()
        temp_mnist = (temp_mnist - temp_mnist.iloc[0]) / (temp_mnist.iloc[-1] - temp_mnist.iloc[0])
        temp_mnist_df = np.all(temp_mnist[temp_mnist >= 0.99].notnull(), axis=1)
        mnist_idx = temp_mnist_df[temp_mnist_df == True].index[0]

        idx_collection = mnist_results.loc[:mnist_idx, "Feature_Importance_Index"]
        modified_data = mnist_X.iloc[:, idx_collection]
        modified_data.reset_index(inplace=True, drop=True)
        modified_data["Label"] = mnist_y
        with open(f"{os.getcwd()}/DimensionalityReduction/RF_MNIST_Reduced_Dataset.pkl", "wb") as output_file:
            pickle.dump(modified_data, output_file)
            output_file.close()

        fashion_results["KNN_ROC"] = fashion_results["KNN_Accuracy"].pct_change(periods=10)
        fashion_results["SVM_ROC"] = fashion_results["SVM_Accuracy"].pct_change(periods=10)
        fashion_results["DIFF"] = np.abs(fashion_results["KNN_ROC"] - fashion_results["SVM_ROC"])

        temp_fashion = fashion_results[["Decision_Tree_Accuracy", "KNN_Accuracy", "SVM_Accuracy"]].copy()
        temp_fashion = (temp_fashion - temp_fashion.iloc[0]) / (temp_fashion.iloc[-1] - temp_fashion.iloc[0])
        temp_fashion_df = np.all(temp_fashion[temp_fashion >= 0.99].notnull(), axis=1)
        fashion_idx = temp_fashion_df[temp_fashion_df == True].index[0]

        idx_collection = fashion_results["Feature_Importance_Index"].iloc[:fashion_idx]
        modified_data = fashion_X.iloc[:, idx_collection]
        modified_data.reset_index(inplace=True, drop=True)
        modified_data["Label"] = fashion_y
        with open(f"{os.getcwd()}/DimensionalityReduction/RF_Fashion_Reduced_Dataset.pkl", "wb") as output_file:
            pickle.dump(modified_data, output_file)
            output_file.close()

        # region Combined
        plt.close("all")
        plot_limit = 400
        fig, ((ax1_1, ax1_2), (ax2_1, ax2_2)) = plt.subplots(2, 2, figsize=(12, 12))

        ax1_1.plot(temp_mnist["Decision_Tree_Accuracy"].iloc[:plot_limit], label="Decision_Tree_Accuracy")
        ax1_1.plot(temp_mnist["KNN_Accuracy"].iloc[:plot_limit], label="KNN_Accuracy")
        ax1_1.plot(temp_mnist["SVM_Accuracy"].iloc[:plot_limit], label="SVM_Accuracy")
        ax1_1.set_title(f"Accuracy Vs N-Features\nMNIST",
                        fontsize=15, weight='bold')
        ax1_1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax1_1.set_xlabel(f"N Components", fontsize=15, weight='heavy')
        ax1_1.set_ylabel(f"Accuracy", fontsize=15, weight='heavy')
        ax1_1.axvline(mnist_idx, color="navy", linestyle="--", alpha=0.5, lw=2,
                      label=f"N Components @ {mnist_idx}")
        ax1_1.legend(loc="best", markerscale=1.1, frameon=True,
                     edgecolor="black", fancybox=True, shadow=True)
        mnist_pixel_importance = np.zeros(shape=(784,))
        for i in range(mnist_results.shape[0]):
            mnist_pixel_importance[
                mnist_results.loc[i, "Feature_Importance_Index"]] = \
                mnist_results.loc[i, "Feature_Importance"] * 100

        ax2_1.plot(temp_fashion["Decision_Tree_Accuracy"].iloc[:plot_limit], label="Decision_Tree_Accuracy")
        ax2_1.plot(temp_fashion["KNN_Accuracy"].iloc[:plot_limit], label="KNN_Accuracy")
        ax2_1.plot(temp_fashion["SVM_Accuracy"].iloc[:plot_limit], label="SVM_Accuracy")
        ax2_1.set_title(f"Accuracy Vs N-Features\nFashion MNIST",
                        fontsize=15, weight='bold')
        ax2_1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax2_1.set_xlabel(f"N Components", fontsize=15, weight='heavy')
        ax2_1.set_ylabel(f"Accuracy", fontsize=15, weight='heavy')
        ax2_1.axvline(fashion_idx, color="navy", linestyle="--", alpha=0.5, lw=2,
                      label=f"N Components @ {fashion_idx}")
        ax2_1.legend(loc="best", markerscale=1.1, frameon=True,
                     edgecolor="black", fancybox=True, shadow=True)
        fashion_pixel_importance = np.zeros(shape=(784,))
        for i in range(fashion_results.shape[0]):
            fashion_pixel_importance[
                fashion_results.loc[i, "Feature_Importance_Index"]] = \
                fashion_results.loc[i, "Feature_Importance"] * 100
        heatmap(np.reshape(mnist_pixel_importance, (28, 28)),
                row_labels=np.arange(0, 28),
                col_labels=np.arange(0, 28),
                ax=ax1_2, cbarlabel="Importance",
                title=f"Pixel Importance\nMNIST",
                folder="DimensionalityReduction/RandomForest",
                filename=f"Random_Forest_MNIST_Pixel_Importance",
                kwargs={"cmap": "inferno"},
                cmap="inferno")
        heatmap(np.reshape(fashion_pixel_importance, (28, 28)),
                row_labels=np.arange(0, 28),
                col_labels=np.arange(0, 28),
                ax=ax2_2, cbarlabel="Importance",
                title=f"Pixel Importance\nFashion-MNIST",
                folder="DimensionalityReduction/RandomForest",
                filename=f"Random_Forest_Fashion_Pixel_Importance",
                kwargs={"cmap": "inferno"},
                cmap="inferno")
        plt.subplots_adjust(hspace=0.25, wspace=0.35)
        plt.savefig(f"{os.getcwd()}/DimensionalityReduction/RandomForest/RF_Accuracy_Combined.png")
        # endregion

        # region Separate

        # region MNIST
        plt.close("all")
        dataset_name = "MNIST"
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[8, 5], figure=fig, wspace=0.15, hspace=0.25)
        ax0 = fig.add_subplot(gs[:, 0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[1, 1])

        ax1.plot(temp_mnist["Decision_Tree_Accuracy"].iloc[:plot_limit], label="Decision_Tree_Accuracy")
        ax1.plot(temp_mnist["KNN_Accuracy"].iloc[:plot_limit], label="KNN_Accuracy")
        ax1.plot(temp_mnist["SVM_Accuracy"].iloc[:plot_limit], label="SVM_Accuracy")
        ax1.set_title(f"Accuracy Vs N-Features\nMNIST",
                      fontsize=8, weight='bold')
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax1.set_xlabel(f"N Components", fontsize=8, weight='heavy')
        ax1.set_ylabel(f"Accuracy", fontsize=8, weight='heavy')
        ax1.axvline(mnist_idx, color="black", linestyle="--", alpha=0.5, lw=2,
                    label=f"N Components @ {mnist_idx}")
        ax1.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True, fontsize=6)
        ax1.xaxis.set_tick_params(labelsize=6)
        ax1.yaxis.set_tick_params(labelsize=6)
        plot_scatterPlot(mnist_results, mnist_X, mnist_y, ax0, mnist_idx, alpha=0.25,
                         label_font_size=14, label_font_weight="heavy")

        mnist_pixel_importance = np.zeros(shape=(784,))
        for i in range(mnist_results.shape[0]):
            mnist_pixel_importance[
                mnist_results.loc[i, "Feature_Importance_Index"]] = \
                mnist_results.loc[i, "Feature_Importance"] * 100

        heatmap(np.reshape(mnist_pixel_importance, (28, 28)),
                row_labels=np.arange(0, 28),
                col_labels=np.arange(0, 28),
                ax=ax2, cbarlabel="Importance",
                folder="DimensionalityReduction/RandomForest",
                filename=f"Random_Forest_MNIST_Pixel_Importance",
                kwargs={"cmap": "inferno"},
                cmap="inferno", title_size=8, axis_size=8, cbar_fontsize=8)
        ax2.xaxis.set_major_formatter(plt.NullFormatter())
        ax2.yaxis.set_major_formatter(plt.NullFormatter())
        plt.savefig(f"{os.getcwd()}/DimensionalityReduction/RandomForest/RF_MNIST_Pair_Plot.png")
        # endregion

        # region Fashion
        plt.close("all")
        dataset_name = "Fashion"
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[8, 5], figure=fig, wspace=0.15, hspace=0.25)
        ax0 = fig.add_subplot(gs[:, 0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[1, 1])

        ax1.plot(temp_fashion["Decision_Tree_Accuracy"].iloc[:plot_limit], label="Decision_Tree_Accuracy")
        ax1.plot(temp_fashion["KNN_Accuracy"].iloc[:plot_limit], label="KNN_Accuracy")
        ax1.plot(temp_fashion["SVM_Accuracy"].iloc[:plot_limit], label="SVM_Accuracy")
        ax1.set_title(f"Accuracy Vs N-Features\nFashion MNIST",
                      fontsize=8, weight='bold')
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
        ax1.set_xlabel(f"N Components", fontsize=8, weight='heavy')
        ax1.set_ylabel(f"Accuracy", fontsize=8, weight='heavy')
        ax1.axvline(fashion_idx, color="black", linestyle="--", alpha=0.5, lw=2,
                    label=f"N Components @ {fashion_idx}")
        ax1.legend(loc="best", markerscale=1.1, frameon=True,
                   edgecolor="black", fancybox=True, shadow=True, fontsize=6)
        ax1.xaxis.set_tick_params(labelsize=6)
        ax1.yaxis.set_tick_params(labelsize=6)
        plot_scatterPlot(fashion_results, fashion_X, fashion_y, ax0, fashion_idx, alpha=0.25,
                         label_font_size=14, label_font_weight="heavy")

        fashion_pixel_importance = np.zeros(shape=(784,))
        for i in range(fashion_results.shape[0]):
            fashion_pixel_importance[
                fashion_results.loc[i, "Feature_Importance_Index"]] = \
                fashion_results.loc[i, "Feature_Importance"] * 100

        heatmap(np.reshape(fashion_pixel_importance, (28, 28)),
                row_labels=np.arange(0, 28),
                col_labels=np.arange(0, 28),
                ax=ax2, cbarlabel="Importance",
                folder="DimensionalityReduction/RandomForest",
                filename=f"Random_Forest_Fashion_Pixel_Importance",
                kwargs={"cmap": "inferno"},
                cmap="inferno", title_size=8, axis_size=8, cbar_fontsize=8)
        ax2.xaxis.set_major_formatter(plt.NullFormatter())
        ax2.yaxis.set_major_formatter(plt.NullFormatter())
        plt.savefig(f"{os.getcwd()}/DimensionalityReduction/RandomForest/RF_Fashion_Pair_Plot.png")
        # endregion

        # endregion
        return
    except Exception as plot_random_forest_results_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'plot_random_forest_results'", plot_random_forest_results_exception)


def extract_data_and_labels(dataframe):
    try:
        return {"train_X": dataframe.drop(columns="Label"),
                "train_y": dataframe["Label"]}
    except Exception as extract_data_and_labels_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'extract_data_and_labels'", extract_data_and_labels_exception)


def plt_confusion_matrix(fashion_clf, fashion_X, fashion_y,
                         cmap=None, directory=None, extra_name="", fmt="d", plot_width=8, plot_height=8,
                         folder="SVM"):
    try:
        from sklearn.metrics import plot_confusion_matrix
        plt.close('all')
        fashion_conf_matrix = plot_confusion_matrix(fashion_clf, fashion_X, fashion_y, cmap=cmap, values_format=fmt)
        plt.savefig(f"{os.getcwd()}/{directory}/Confusion_Matrix/{extra_name}_Fashion_MNIST_Confusion_Matrix.png")
        return plt

    except Exception as e:
        print(f"Exception in plot_combined_confusion_matrix:\n", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def plot_precision_recall(classifier, trainX, trainY, testX, testY, training_sizes=np.linspace(0.05, 1.00, 20),
                          folder="NeuralNetwork", n_classes=10, plot_title=None, dataset_name='MNIST', plot_width=12,
                          plot_height=6, is_final=False, cv=5, pre_fit=False, is_one_hot=False):
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
            if not pre_fit:
                print("Shit, we should not have refit this...")
                classifier.fit(temp_X, temp_y)

            y_pred = classifier.predict(testX)
            if is_one_hot:
                y_pred = np.argmax(y_pred, axis=1)
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
        plt.savefig(
            f"{os.getcwd()}/figures/{folder}/Metrics/_{folder}_Precision_{dataset_name}_{dataset_name}_{extra}.png",
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
        plt.savefig(f"{os.getcwd()}/figures/{folder}/Metrics/_{folder}_Recall_{dataset_name}_{extra}.png",
                    bbox_inches='tight')

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
        plt.savefig(f"{os.getcwd()}/figures/{folder}/Metrics/_{folder}_F1Score_{dataset_name}_{extra}.png",
                    bbox_inches='tight')
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
            plt.savefig(
                f"{os.getcwd()}/figures/{folder}/Metrics/_{folder}_Combined_Precision_Recall_{dataset_name}_{extra}.png",
                bbox_inches='tight')

        return precisions, recalls, f1_score

    except Exception as e:
        print(f"Exception in plot_combined_complexity:\n", e)
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


def get_learning_curve(classifier, train_X, train_y, train_sizes=np.linspace(0.1, 1.0, 5), cv=5,
                       n_jobs=-1, verbose=1, random_number=SEED, return_times=False, backend='threading',
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


def plot_learning_curve_1(estimator, title, train_X, train_y, axes=None, ylim=(0.6, 1.01), cv=None,
                          f_name="My_Plot",
                          n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), save_individual=False,
                          TESTING=False, backend='loky', extra_name="", folder="SVM", confusion=False,
                          confusion_name="MNIST", pre_fit=False):
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
                               random_state=SEED,
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
            plt.savefig(f"{os.getcwd()}/DimensionalityReduction/NeuralNetwork/_{extra_name}_Learning_Curve.png",
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
            plt.savefig(f"{os.getcwd()}/DimensionalityReduction/NeuralNetwork/_{extra_name}_Fit_Times.png",
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
            plt.savefig(f"{os.getcwd()}/DimensionalityReduction/NeuralNetwork/_{extra_name}_Fit_Times_Vs_Score.png",
                        bbox_inches='tight')
        plt.close("all")
        if confusion:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            # Fit Estimator
            print(f"Starting Confusion Matrix")
            temp_estimator = copy.deepcopy(estimator)
            if not pre_fit:
                temp_estimator.fit(train_X, train_y)

            plot_confusion_matrix(estimator=temp_estimator, X=train_X, y_true=train_y, cmap=None,
                                  values_format="d", ax=ax2)
            ax2.set_title(f"{confusion_name} \nConfusion Matrix", fontsize=15, weight='bold')
            ax2.set_xlabel("Predicted Label", fontsize=15, weight='heavy')
            ax2.set_ylabel("True Label", fontsize=15, weight='heavy')

            ax1.set_title(title, fontsize=15, weight='bold')
            ylim = (0.6, 1.01)
            ax1.set_xlabel("Training examples", fontsize=15, weight='heavy')
            ax1.set_ylabel("Score", fontsize=15, weight='heavy')
            ax1.set_ylim(ylim[0], ylim[1])

            # Customize the major grid
            ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')

            ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.2,
                             color="navy")
            ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.2,
                             color="darkorange")
            ax1.plot(train_sizes, train_scores_mean, 'o-', color="navy",
                     label="Training score")
            ax1.plot(train_sizes, test_scores_mean, 'o-', color="darkorange",
                     label="Cross-validation score")
            ax1.set_ylim(ylim[0], ylim[1])
            ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
            check_folder(_directory=f"{folder}/Confusion_Matrix")
            plt.tight_layout()
            plt.savefig(f"{os.getcwd()}/_{extra_name}_{confusion_name}_Confusion_Matrix.png")

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
        plt.savefig(f"{os.getcwd()}/DimensionalityReduction/NeuralNetwork/_{extra_name}_Combined_.png",
                    bbox_inches='tight')
        return temp_df, results

    except Exception as e:
        print(f"Exception in plot_learning_curve:\n", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def test():
    VERBOSE = 0
    temp_folder = "/Clustering/" + "Reduced/"
    check_folder(temp_folder)
    save_dir = os.getcwd() + temp_folder
    limit = 5000
    idx = [i for i in range(2, 32, 1)]
    cols = ["Inertia", "Silhouette", "Homogeneity", "Completeness", "Harmonic_Mean", "Calinski_Harabasz",
            "Davies_Bouldin"]
    mnist_results = pd.DataFrame(columns=cols, index=idx,
                                 data=np.zeros(shape=(len(idx), len(cols))))

    fashion_results = pd.DataFrame(columns=cols, index=idx,
                                   data=np.zeros(shape=(len(idx), len(cols))))
    algorithm_names = ["PCA", "ICA", "RP", "RF"]
    count = 0
    print("Starting K-Means Clustering")
    for _df in ["MNIST", "Fashion-MNIST"]:
        for alg in algorithm_names:
            for k in idx:
                if _df == "MNIST":
                    filename = f"{os.getcwd()}/DimensionalityReduction/{alg}_{_df}_Reduced_Dataset.pkl"
                    with open(filename, "rb") as input_file:
                        temp_reduced = pickle.load(input_file)
                        input_file.close()
                    data = extract_data_and_labels(temp_reduced)
                    temp_train_X = data["train_X"]
                    temp_train_y = data["train_y"]
                    print(f"Current Dataset: {_df}")
                    print(f"\tCurrent Algorithm: {alg}")
                    print(f"\tTrain Data Shape: {temp_train_X.shape}")
                    print()
                    k_means = KMeans(n_clusters=k, verbose=VERBOSE).fit(temp_train_X)
                    inertia = k_means.inertia_
                    silhouette_average = silhouette_score(temp_train_X, k_means.labels_, sample_size=limit)
                    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(temp_train_y,
                                                                                              k_means.labels_)
                    mnist_results.loc[k, "Inertia"] = inertia
                    mnist_results.loc[k, "Silhouette"] = silhouette_average
                    mnist_results.loc[k, "Calinski_Harabasz"] = calinski_harabasz_score(temp_train_X, k_means.labels_)
                    mnist_results.loc[k, "Davies_Bouldin"] = davies_bouldin_score(temp_train_X, k_means.labels_)
                    mnist_results.loc[k, "Homogeneity"] = homogeneity
                    mnist_results.loc[k, "Completeness"] = completeness
                    mnist_results.loc[k, "Harmonic_Mean"] = v_measure
                    if count >= 5:
                        count = 0
                        with open(f"{save_dir}/{_df}_{alg}_Results.pkl", "wb") as output_file:
                            pickle.dump(mnist_results, output_file)
                            output_file.close()
                    print(f"\n\t{_df} - k={k} \n{mnist_results.loc[k]}")
                    results = mnist_results
                elif _df == "Fashion-MNIST":
                    filename = f"{os.getcwd()}/DimensionalityReduction/{alg}_Fashion_Reduced_Dataset.pkl"
                    with open(filename, "rb") as input_file:
                        temp_reduced = pickle.load(input_file)
                        input_file.close()
                    data = extract_data_and_labels(temp_reduced)
                    temp_train_X = data["train_X"]
                    temp_train_y = data["train_y"]
                    print(f"Current Dataset: {_df}")
                    print(f"\tCurrent Algorithm: {alg}")
                    print(f"\tTrain Data Shape: {temp_train_X.shape}")
                    k_means = KMeans(n_clusters=k, verbose=VERBOSE).fit(temp_train_X)
                    inertia = k_means.inertia_
                    silhouette_average = silhouette_score(temp_train_X, k_means.labels_, sample_size=limit)
                    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(temp_train_y,
                                                                                              k_means.labels_)
                    fashion_results.loc[k, "Inertia"] = inertia
                    fashion_results.loc[k, "Silhouette"] = silhouette_average
                    fashion_results.loc[k, "Calinski_Harabasz"] = calinski_harabasz_score(temp_train_X, k_means.labels_)
                    fashion_results.loc[k, "Davies_Bouldin"] = davies_bouldin_score(temp_train_X, k_means.labels_)
                    fashion_results.loc[k, "Homogeneity"] = homogeneity
                    fashion_results.loc[k, "Completeness"] = completeness
                    fashion_results.loc[k, "Harmonic_Mean"] = v_measure
                    if count >= 5:
                        count = 0
                        with open(f"{save_dir}/{_df}_{alg}_Results.pkl", "wb") as output_file:
                            pickle.dump(fashion_results, output_file)
                            output_file.close()
                    print(f"\n\t{_df} - k={k} \n{fashion_results.loc[k]}")
                    results = fashion_results
            with open(f"{save_dir}/{_df}_{alg}_Results.pkl", "wb") as output_file:
                pickle.dump(results, output_file)
                output_file.close()
    return


def part_4(dataset_name):
    try:
        temp_folder = "/DimensionalityReduction/" + "NeuralNetwork/"
        check_folder(temp_folder)
        algorithm_names = ["PCA", "ICA", "RP", "RF"]
        all_data = {}
        res = {}
        for alg in algorithm_names:
            filename = f"{os.getcwd()}/DimensionalityReduction/{alg}_{dataset_name}_Reduced_Dataset.pkl"
            with open(filename, "rb") as input_file:
                temp_reduced = pickle.load(input_file)
                input_file.close()
            all_data[f"{alg}"] = extract_data_and_labels(temp_reduced)

            temp_train_X = all_data[f"{alg}"]["train_X"]
            temp_train_y = all_data[f"{alg}"]["train_y"]
            title = f"{alg} {dataset_name}\n Learning Curve"
            train_sizes = np.linspace(0.05, 1.0, 20)
            f_name = f"{alg}_{dataset_name}"
            res['nn_results'], res['cv_results'] = plot_learning_curve_1(
                estimator=MLPClassifier(hidden_layer_sizes=(40,),
                                        max_iter=1000,
                                        random_state=42,
                                        verbose=False,
                                        warm_start=True),
                title=title, train_X=temp_train_X,
                train_y=temp_train_y, cv=5,
                folder="NeuralNetwork", f_name=f_name,
                train_sizes=train_sizes,
                extra_name=alg,
                save_individual=True, TESTING=True, backend='loky')
            print()

        return
    except Exception as part_4_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'part_4'", part_4_exception)


def part_5():
    try:
        print()
    except Exception as part_5_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'part_5'", part_5_exception)


def plot_better_results(data_X, data_y, data_results, dataset_name, algorithm_name, vline_idx, pixel_importance, model,
                        original_data_X, original_data_y, results, is_rand_proj=False, is_ica=False, is_pca=False,
                        is_rand_forest=False, font_size=14, is_fashion=False, n_clusters=10, use_tsne=False):
    try:
        fashion_labels = {0: "T-shirt/top",
                          1: "Trouser",
                          2: "Pullover",
                          3: "Dress",
                          4: "Coat",
                          5: "Sandal",
                          6: "Shirt",
                          7: "Sneaker",
                          8: "Bag",
                          9: "Ankle Boot"}
        rand_idx = np.random.randint(0, 784)
        reconstructed_label = original_data_y.iloc[rand_idx]
        if is_fashion:
            reconstructed_label = fashion_labels[original_data_y.iloc[rand_idx]]
        original = np.reshape(original_data_X.iloc[rand_idx, :].to_numpy(), (28, 28))
        reconstructed_img = original
        plt.close("all")

        # region Separate

        plt.close("all")
        fig = plt.figure(figsize=(16, 8), constrained_layout=True)
        gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[9, 6, 6], figure=fig)
        large_img_on_left = fig.add_subplot(gs[:, 0])
        top_middle_img = fig.add_subplot(gs[0, 1])
        bottom_middle_img = fig.add_subplot(gs[1, 1])
        top_right_img = fig.add_subplot(gs[0, 2])
        bottom_right_img = fig.add_subplot(gs[1, 2])

        if is_rand_proj:
            random_proj = GaussianRandomProjection(n_components=vline_idx).fit(original_data_X)
            transformed_data = pd.DataFrame(random_proj.transform(original_data_X))
            inverse = pd.DataFrame(np.linalg.pinv(random_proj.components_.T))
            reconstructed_data = transformed_data.dot(inverse)
            tmp = reconstructed_data.iloc[rand_idx, :].to_numpy()
            reconstructed_img = np.reshape(tmp, (28, 28))
            limit_for_ten_percent_error = 0.01
            top_middle_img.plot(data_results, label="Avg Reconstruction Error")
            top_middle_img.set_title(f"Reconstruction Error\n{dataset_name}",
                          fontsize=font_size, weight='bold')
            top_middle_img.grid(which='major', linestyle='-', linewidth='0.5', color='white')
            top_middle_img.set_xlabel(f"N Components", fontsize=font_size, weight='heavy')
            top_middle_img.set_ylabel(f"Reconstruction Error", fontsize=font_size, weight='heavy')
            top_middle_img.axvline(vline_idx, color="navy", linestyle="--", alpha=0.5, lw=2,
                        label=f"Number of Components @ {vline_idx}")
            top_middle_img.axhline(limit_for_ten_percent_error, color="black", linestyle="--", alpha=0.5, lw=2,
                        label=f"Threshold of 10% Reconstruction Error")
            top_middle_img.legend(loc="best", markerscale=1.1, frameon=True,
                       edgecolor="black", fancybox=True, shadow=True)

        elif is_rand_forest:
            top_middle_img.plot(data_results["Decision_Tree_Accuracy"], label="Decision_Tree_Accuracy")
            top_middle_img.plot(data_results["KNN_Accuracy"], label="KNN_Accuracy")
            top_middle_img.plot(data_results["SVM_Accuracy"], label="SVM_Accuracy")
            top_middle_img.set_title(f"Accuracy Vs N-Features\nMNIST",
                            fontsize=font_size, weight='bold')
            top_middle_img.grid(which='major', linestyle='-', linewidth='0.5', color='white')
            top_middle_img.set_xlabel(f"N Components", fontsize=font_size, weight='heavy')
            top_middle_img.set_ylabel(f"Accuracy", fontsize=font_size, weight='heavy')
            top_middle_img.axvline(vline_idx, color="navy", linestyle="--", alpha=0.5, lw=2,
                          label=f"N Components @ {vline_idx}")
            top_middle_img.legend(loc="best", markerscale=1.1, frameon=True,
                         edgecolor="black", fancybox=True, shadow=True)
            pixel_importance = np.zeros(shape=(784,))
            for i in range(data_results.shape[0]):
                pixel_importance[
                    data_results.loc[i, "Feature_Importance_Index"]] = \
                    data_results.loc[i, "Feature_Importance"] * 100

        elif is_ica:
            transformed_dataset = model.transform(original_data_X)
            recon_X = model.inverse_transform(transformed_dataset)[rand_idx]
            reconstructed_img = np.reshape(recon_X, (28, 28))
            top_middle_img.set_xlabel(f"N Components", fontsize=font_size, weight='heavy')
            top_middle_img.set_ylabel(f"Kurtosis/Reconstruction Error", fontsize=font_size, weight='heavy')
            ln1 = top_middle_img.plot(results["Avg_Kurtosis"].index.values, results["Avg_Kurtosis"], label="Average Kurtosis")
            top_middle_img.set_title(f"Kurtosis Vs. Reconstruction Error\n{dataset_name}",
                            fontsize=font_size, weight='bold')
            top_middle_img.grid(which='major', linestyle='-', linewidth='0.5', color='white')
            ln3 = top_middle_img.axvline(vline_idx, color="black", linestyle="--", alpha=0.5, lw=2,
                                label=f"Max_Kurtosis @ {vline_idx:.2f}")
            top_middle_img_secondary = top_middle_img.twinx()
            top_middle_img_secondary.set_xlabel(f"N Components", fontsize=font_size, weight='heavy')
            ln2 = top_middle_img_secondary.plot(results["Avg_Reconstruction_Error"],
                                       label="Average Reconstruction Error", color="navy")
            top_middle_img_secondary.annotate("Error @ 1%",
                                     (vline_idx + 1,
                                      results.loc[vline_idx, "Avg_Reconstruction_Error"]),
                                     fontsize=8, weight='heavy')
            top_middle_img_secondary.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.05)
            lns = ln1 + ln2
            lns.append(ln3)
            labs = [l.get_label() for l in lns]
            top_middle_img.legend(lns, labs, loc="best", markerscale=1.1, frameon=True,
                         edgecolor="black", fancybox=True, shadow=True, fontsize=np.ceil(font_size/2))
        elif is_pca:
            top_middle_img.plot(data_results["Variance_DF"]["CumSum"], label="Explained Variance")
            recon_data = pd.DataFrame(model.transform(original_data_X))
            pred_X = recon_data.iloc[rand_idx]
            reconstructed_img = np.reshape(model.inverse_transform(pred_X), (28, 28))
            top_middle_img.axhline(data_results["Kept_Variance"], color="black", linestyle="--", alpha=0.5, lw=2,
                          label=f"Explained Variance @ {data_results['Kept_Variance']:.3f}")

            top_middle_img.axvline(data_results["Number_Of_Components_To_Keep"], color="navy", linestyle="--", alpha=0.5, lw=2,
                          label=f"Number of Components @ {data_results['Number_Of_Components_To_Keep']}")

            top_middle_img.set_title(f"Explained Variance Vs. Number of Components\n{dataset_name}",
                            fontsize=15, weight='bold')
            top_middle_img.grid(which='major', linestyle='-', linewidth='0.5', color='white')
            top_middle_img.set_xlabel(f"N Components", fontsize=15, weight='heavy')
            top_middle_img.set_ylabel(f"Explained Variance", fontsize=15, weight='heavy')
            top_middle_img.legend(loc="best", markerscale=1.1, frameon=True,
                         edgecolor="black", fancybox=True, shadow=True, fontsize=np.ceil(font_size/2))

        top_middle_img.tick_params(axis="both")
        top_right_img.imshow(original, cmap="gray")
        top_right_img.set_ylabel(f"Label={reconstructed_label}", fontsize=font_size, weight='heavy')
        top_right_img.yaxis.set_label_position("right")
        top_right_img.set_title(f"Original Data",
                      fontsize=font_size, weight='bold')
        top_right_img.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)

        bottom_right_img.imshow(reconstructed_img, cmap="gray")
        bottom_right_img.set_ylabel(f"Label={reconstructed_label}", fontsize=font_size, weight='heavy')
        bottom_right_img.yaxis.set_label_position("right")
        bottom_right_img.set_title(f"Reconstructed Data", fontsize=font_size, weight='bold')

        bottom_right_img.grid(which='major', linestyle='-', linewidth='0.5', color='white', alpha=0.2)
        plot_scatterPlot(data_results, data_X, data_y, large_img_on_left, vline_idx, alpha=0.75,
                         label_font_size=2*font_size, label_font_weight="heavy", n_clusters=n_clusters)
        if is_pca:
            heatmap(np.reshape(data_results["Results"].mean_, (28, 28)),
                    row_labels=np.arange(0, 28),
                    col_labels=np.arange(0, 28),
                    ax=bottom_middle_img, cbarlabel="Importance",
                    folder="DimensionalityReduction/PrincipalComponentAnalysis",
                    filename=f"{algorithm_name}_{dataset_name}_Pixel_Importance",
                    kwargs={"cmap": "inferno"},
                    cmap="inferno", title_size=font_size, axis_size=font_size, cbar_fontsize=font_size)
        elif is_rand_proj:
            heatmap(reconstructed_img,
                    row_labels=np.arange(0, 28),
                    col_labels=np.arange(0, 28),
                    ax=bottom_middle_img, cbarlabel="Importance",
                    folder="DimensionalityReduction/PrincipalComponentAnalysis",
                    filename=f"{algorithm_name}_{dataset_name}_Pixel_Importance",
                    kwargs={"cmap": "inferno"},
                    cmap="inferno", title_size=font_size, axis_size=font_size, cbar_fontsize=font_size)
        else:
            heatmap(np.reshape(pixel_importance, (28, 28)),
                    row_labels=np.arange(0, 28),
                    col_labels=np.arange(0, 28),
                    ax=bottom_middle_img, cbarlabel="Importance",
                    folder="DimensionalityReduction/RandomForest",
                    filename=f"{algorithm_name}_{dataset_name}_Pixel_Importance",
                    kwargs={"cmap": "inferno"},
                    cmap="inferno", title_size=font_size, axis_size=font_size, cbar_fontsize=font_size)
        large_img_on_left.set_title(f"Dimensionality Reduced Data", fontsize=font_size, weight='bold')
        large_img_on_left.set_ylabel(f"y", fontsize=font_size, weight='heavy')
        large_img_on_left.set_xlabel(f"X", fontsize=font_size, weight='heavy')
        plt.savefig(f"{os.getcwd()}/DimensionalityReduction/{algorithm_name}_{dataset_name}_Pair_Plot.png")

        # endregion
        return
    except Exception as plot_random_forest_results_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'plot_random_forest_results'", plot_random_forest_results_exception)

