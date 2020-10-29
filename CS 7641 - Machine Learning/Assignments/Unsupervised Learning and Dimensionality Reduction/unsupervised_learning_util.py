import glob
import os
import pickle
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from scipy.stats import kurtosis
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
    if folder is None:
        plt.savefig(f"{filename}.png")
    else:
        plt.savefig(f"{folder}/{filename}.png")
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
                     label_font_size=14, label_font_weight="heavy", cmap="tab10"):
    try:
        idx_collection = results["Feature_Importance_Index"].iloc[:idx].to_numpy()
        modified_data = data_X.iloc[:, idx_collection]
        temp_reducer = TSNE()
        reduced_modified_data = temp_reducer.fit_transform(modified_data)

        k_cluster = KMeans(n_clusters=10)
        clustered_data = k_cluster.fit(reduced_modified_data)

        temp_DF = pd.DataFrame(columns=["X", "y"], data=reduced_modified_data)
        temp_DF["Label"] = clustered_data.labels_
        temp_DF.plot.scatter("X", "y", c="Label", cmap=cmap, ax=ax, alpha=alpha, edgecolors='black')

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
                     mnist_X_scaled, mnist_y_scaled, fashion_X_scaled, fashion_y_scaled):
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
        max_kurtosis_mnist = np.argmax(mnist_df["Avg_Kurtosis"])
        # Getting image b/c I forgot to save data
        temp_results = FastICA(n_components=max_kurtosis_mnist, whiten=True).fit(mnist_X_scaled)
        transformed_mnist_dataset = temp_results.transform(mnist_X_scaled)
        with open(f"{os.getcwd()}/DimensionalityReduction/ICA_MNIST_Reduced_Dataset.pkl", "wb") as output_file:
            pickle.dump(transformed_mnist_dataset, output_file)
            output_file.close()
        image = np.reshape(temp_results.mean_, (28, 28))
        heatmap(image,
                row_labels=np.arange(0, 28),
                col_labels=np.arange(0, 28),
                ax=ax1_2, cbarlabel="Importance",
                title=f"MNIST\nPixel Importance",
                folder="DimensionalityReduction/IndependentComponentAnalysis",
                filename=f"Mnist_Pixel_Importance",
                kwargs={"cmap": "inferno"},
                cmap="inferno")
        ln3 = ax1_1.axvline(max_kurtosis_mnist, color="black", linestyle="--", alpha=0.5, lw=2,
                            label=f"Max_Kurtosis @ {max_kurtosis_mnist:.2f}")

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
        max_kurtosis_fashion = np.argmax(fashion_df["Avg_Kurtosis"])
        # Getting image b/c I forgot to save data
        temp_results = FastICA(n_components=max_kurtosis_fashion, whiten=True).fit(fashion_X_scaled)
        transformed_fashion_dataset = temp_results.transform(fashion_X_scaled)
        with open(f"{os.getcwd()}/DimensionalityReduction/ICA_Fashion_Reduced_Dataset.pkl", "wb") as output_file:
            pickle.dump(transformed_fashion_dataset, output_file)
            output_file.close()
        image = np.reshape(temp_results.mean_, (28, 28))
        heatmap(image,
                row_labels=np.arange(0, 28),
                col_labels=np.arange(0, 28),
                ax=ax2_2, cbarlabel="Importance",
                title=f"Fashion MNIST\nPixel Importance",
                folder="DimensionalityReduction/IndependentComponentAnalysis",
                filename=f"Fashion_Mnist_Pixel_Importance",
                kwargs={"cmap": "inferno"},
                cmap="inferno")
        ln3 = ax2_1.axvline(max_kurtosis_fashion, color="black", linestyle="--", alpha=0.5, lw=2,
                            label=f"Max_Kurtosis @ {max_kurtosis_fashion:.2f}")

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
        max_kurtosis_mnist = np.argmax(mnist_df["Avg_Kurtosis"])
        # Getting image b/c I forgot to save data
        temp_results = FastICA(n_components=max_kurtosis_mnist, whiten=True).fit(mnist_X_scaled)
        transformed_mnist_dataset = temp_results.transform(mnist_X_scaled)
        with open(f"{os.getcwd()}/DimensionalityReduction/ICA_MNIST_Reduced_Dataset.pkl", "wb") as output_file:
            pickle.dump(transformed_mnist_dataset, output_file)
            output_file.close()
        image = np.reshape(temp_results.mean_, (28, 28))
        heatmap(image,
                row_labels=np.arange(0, 28),
                col_labels=np.arange(0, 28),
                ax=ax1_2, cbarlabel="Importance",
                title=f"Pixel Importance\n{dataset_name}",
                folder="DimensionalityReduction/IndependentComponentAnalysis",
                filename=f"Mnist_Pixel_Importance",
                kwargs={"cmap": "inferno"},
                cmap="inferno")
        ln3 = ax1_1.axvline(max_kurtosis_mnist, color="black", linestyle="--", alpha=0.5, lw=2,
                            label=f"Max_Kurtosis @ {max_kurtosis_mnist:.2f}")

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
        recon_X = temp_results.inverse_transform(transformed_mnist_dataset)[rand_idx]

        ax2_2.imshow(np.reshape(recon_X, (28, 28)), cmap="gray")
        ax2_1.imshow(np.reshape(mnist_X.iloc[rand_idx].to_numpy(), (28, 28)), cmap="gray")

        ax2_1.set_title(f"Training Set\nMNIST",
                        fontsize=15, weight='bold')
        ax2_2.set_title(f"Feature Spaced Reduced by "
                        f"{784 - max_kurtosis_mnist}\n"
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
        max_kurtosis_fashion = np.argmax(mnist_df["Avg_Kurtosis"])
        # Getting image b/c I forgot to save data
        temp_results = FastICA(n_components=max_kurtosis_fashion, whiten=True).fit(fashion_X_scaled)
        transformed_fashion_dataset = temp_results.transform(fashion_X_scaled)
        with open(f"{os.getcwd()}/DimensionalityReduction/ICA_Fashion_Reduced_Dataset.pkl", "wb") as output_file:
            pickle.dump(transformed_fashion_dataset, output_file)
            output_file.close()
        image = np.reshape(temp_results.mean_, (28, 28))
        heatmap(image,
                row_labels=np.arange(0, 28),
                col_labels=np.arange(0, 28),
                ax=ax1_2, cbarlabel="Importance",
                title=f"Pixel Importance\n{dataset_name}",
                folder="DimensionalityReduction/IndependentComponentAnalysis",
                filename=f"{dataset_name}_Pixel_Importance",
                kwargs={"cmap": "inferno"},
                cmap="inferno")
        ln3 = ax1_1.axvline(max_kurtosis_fashion, color="black", linestyle="--", alpha=0.5, lw=2,
                            label=f"Max_Kurtosis @ {max_kurtosis_fashion:.2f}")

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
        recon_X = temp_results.inverse_transform(transformed_fashion_dataset)[rand_idx]

        ax2_2.imshow(np.reshape(recon_X, (28, 28)), cmap="gray")
        ax2_1.imshow(np.reshape(fashion_X.iloc[rand_idx].to_numpy(), (28, 28)), cmap="gray")

        ax2_1.set_title(f"Training Set\n{dataset_name}",
                        fontsize=15, weight='bold')
        ax2_2.set_title(f"Feature Spaced Reduced by "
                        f"{784 - max_kurtosis_fashion}\n"
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


def plot_randomized_projection_results(mnist_results, fashion_results, mnist_X, mnist_y, fashion_X, fashion_y,
                                       mnist_X_scaled, mnist_y_scaled, fashion_X_scaled, fashion_y_scaled):
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

        random_proj = GaussianRandomProjection(n_components=mnist_component).fit(mnist_X_scaled)
        transformed_data = pd.DataFrame(random_proj.transform(mnist_X_scaled))
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

        fashion_results["KNN_ROC"] = fashion_results["KNN_Accuracy"].pct_change(periods=10)
        fashion_results["SVM_ROC"] = fashion_results["SVM_Accuracy"].pct_change(periods=10)
        fashion_results["DIFF"] = np.abs(fashion_results["KNN_ROC"] - fashion_results["SVM_ROC"])

        temp_fashion = fashion_results[["Decision_Tree_Accuracy", "KNN_Accuracy", "SVM_Accuracy"]].copy()
        temp_fashion = (temp_fashion - temp_fashion.iloc[0]) / (temp_fashion.iloc[-1] - temp_fashion.iloc[0])
        temp_fashion_df = np.all(temp_fashion[temp_fashion >= 0.99].notnull(), axis=1)
        fashion_idx = temp_fashion_df[temp_fashion_df == True].index[0]

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
