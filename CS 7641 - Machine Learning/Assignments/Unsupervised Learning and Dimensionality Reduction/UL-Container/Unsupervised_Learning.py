import os
import sys

import pandas as pd
import numpy as np
import pickle
import unsupervised_learning_util as utl
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS, Isomap, SpectralEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from warnings import simplefilter
from yellowbrick.cluster import KElbowVisualizer

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500

plt.tight_layout()

simplefilter(action='ignore', category=FutureWarning)
TESTING = False
NJOBS = 32
VERBOSE = 0
N_Clusters = 30


def calculate_reconstruction_error():
    try:
        print()
    except Exception as calculate_reconstruction_error_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'calculate_reconstruction_error'", calculate_reconstruction_error_exception)


def reconstruct_from_reduced():
    try:
        print()
    except Exception as reconstruct_from_reduced_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'reconstruct_from_reduced'", reconstruct_from_reduced_exception)


def run_tsne(dataset, njobs=20, folder="TSNE", dataset_name="MNIST"):
    try:
        temp_folder = folder + "TSNE"
        utl.check_folder(temp_folder)
        return
    except Exception as run_tsne_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'run_tsne'", run_tsne_exception)


def run_pca(dataset, njobs=20, folder="PrincipleComponentAnalysis", dataset_name="MNIST"):
    try:
        temp_folder = folder + "PrincipleComponentAnalysis"
        utl.check_folder(temp_folder)
        return
    except Exception as run_pca_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'run_pca'", run_pca_exception)


def run_random_pca(dataset, njobs=20, folder="RandomPrincipleComponentAnalysis"):
    try:
        temp_folder = folder + "RandomPrincipleComponentAnalysis"
        utl.check_folder(temp_folder)
    except Exception as run_random_pca_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'run_random_pca'", run_random_pca_exception)


def run_ica(dataset, njobs=20, folder="IndependentComponentAnalysis", dataset_name="MNIST"):
    try:
        temp_folder = folder + "IndependentComponentAnalysis"
        utl.check_folder(temp_folder)
        return
    except Exception as run_ica_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'run_ica'", run_ica_exception)


def run_random_projections(dataset, njobs=20, folder="RandomProjections", dataset_name="MNIST"):
    try:
        temp_folder = folder + "RandomProjections"
        utl.check_folder(temp_folder)
        return
    except Exception as run_random_projections_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'run_random_projections'", run_random_projections_exception)


def run_k_means_clustering(dataset, max_clusters=10, n_jobs=20, folder="Kmeans", dataset_name="MNIST",
                           incremental=False, step=5):
    try:
        temp_folder = folder + "KMeans/"
        utl.check_folder(temp_folder)
        save_dir = os.getcwd() + temp_folder
        if incremental:
            cols = [i for i in range(step, max_clusters+1, step)]
        else:
            cols = [i for i in range(2, max_clusters)]
        index = [1000, 5000]
        if max_clusters > index[0]:
            raise ValueError
        inertia_results = pd.DataFrame(columns=cols,
                                       index=index,
                                       data=np.zeros(shape=(len(index), len(cols))))

        silhouette_average_results = pd.DataFrame(columns=cols, index=index,
                                                  data=np.zeros(shape=(len(index), len(cols))))
        silhouette_sample_results = {}
        best_inertia = -1e12
        best_inertia_idx = 0
        best_inertia_num_cluster = 0
        best_silhouette = 0
        best_silhouette_idx = 0
        best_silhouette_num_cluster = 0
        print("Starting K-Means Clustering")
        for idx in index:
            print(f"\t\tNumber of samples: {idx}")
            for num_cluster in cols:
                print(f"\t\tNumber of Clusters: {num_cluster}")
                temp_train_X = dataset["train_X"].iloc[:idx, :]
                k_means = KMeans(n_clusters=num_cluster, precompute_distances=True,
                                 n_jobs=NJOBS, verbose=VERBOSE).fit(temp_train_X)
                inertia = k_means.inertia_
                inertia_results.loc[idx, num_cluster] = inertia
                silhouette_average = silhouette_score(temp_train_X, k_means.labels_)
                silhouette_average_results.loc[idx, num_cluster] = silhouette_average
                temp_silhouette_sample_results = silhouette_samples(temp_train_X, k_means.labels_)
                silhouette_sample_results[f"NumClusters_{num_cluster} DataSize_{idx}"] = temp_silhouette_sample_results
                if inertia > best_inertia:
                    print(f"Best Inertia thus far: {best_inertia}")
                    print(f"\tCurrent Inertia: {inertia}")
                    best_inertia = inertia
                    best_inertia_idx = idx
                    best_inertia_num_cluster = num_cluster
                    print(f"\tNew Best Inertia: {best_inertia}")
                    print(f"\t\tInertia Best IDX: {best_inertia_idx}")
                    print(f"\t\tInertia Best Number of Clusters: {best_inertia_num_cluster}")
                if silhouette_average > best_silhouette:
                    print(f"Best Silhouette thus far: {best_silhouette}")
                    print(f"\tCurrent Silhouette: {silhouette_average}")
                    best_silhouette = silhouette_average
                    best_silhouette_idx = idx
                    best_silhouette_num_cluster = num_cluster
                    print(f"\tNew Best Silhouette: {best_silhouette}")
                    print(f"\t\tSilhouette Best IDX: {best_silhouette_idx}")
                    print(f"\t\tSilhouette Best Number of Clusters: {best_silhouette_num_cluster}")
        with open(f"{save_dir}Inertia_DF_Results_{dataset_name}.pkl", 'wb') as save_file:
            pickle.dump(inertia_results, save_file)
            save_file.close()
        with open(f"{save_dir}Silhouette_DF_Results_{dataset_name}.pkl", 'wb') as save_file:
            pickle.dump(silhouette_average_results, save_file)
            save_file.close()
        with open(f"{save_dir}Silhouette_Sample_Results_{dataset_name}.pkl", 'wb') as save_file:
            pickle.dump(silhouette_sample_results, save_file)
            save_file.close()
        return inertia_results, silhouette_average_results
    except Exception as run_k_means_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'run_k_means_clustering'", run_k_means_exception)


def run_expectation_maximization(dataset, n_jobs=20, folder="ExpectationMaximization",
                                 dataset_name="MNIST"):
    try:
        temp_folder = folder + "ExpectationMaximization"
        utl.check_folder(temp_folder)
        return
    except Exception as run_expectation_maximization_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'run_expectation_maximization'", run_expectation_maximization_exception)


def get_max_components(data):
    try:
        return data.shape[1]
    except Exception as get_max_components_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'get_max_components'", get_max_components_exception)


def run_clustering(mnist, fashion_mnist, run_check=None, scaled=False):
    try:
        folder = "/Clustering/"
        utl.check_folder(folder)
        extra = "NotScaled"
        if scaled:
            extra = "Scaled"
        if run_check is not None:
            if run_check["Run K-Means Clustering"]:
                inertia_mnist, silhouette_mnist = run_k_means_clustering(mnist, max_clusters=N_Clusters, folder=folder,
                                                                         dataset_name=f"MNIST_{extra}")
            if run_check["Run Expectation Maximization"]:
                run_expectation_maximization(mnist, folder=folder, dataset_name=f"MNIST_{extra}")
            if not TESTING:
                if run_check["Run K-Means Clustering"]:
                    inertia_fashion_mnist, \
                    silhouette_fashion_mnist = run_k_means_clustering(fashion_mnist, max_clusters=N_Clusters,
                                                                      folder=folder,
                                                                      dataset_name=f"Fashion_MNIST_{extra}")
                if run_check["Run Expectation Maximization"]:
                    run_expectation_maximization(fashion_mnist, folder=folder, dataset_name=f"Fashion_MNIST_{extra}")
        else:
            inertia_mnist, silhouette_mnist = run_k_means_clustering(mnist, max_clusters=N_Clusters,
                                                                     folder=folder,
                                                                     dataset_name=f"MNIST_{extra}")
            run_expectation_maximization(mnist, folder=folder, dataset_name=f"MNIST_{extra}")
            if not TESTING:
                inertia_fashion_mnist, \
                silhouette_fashion_mnist = run_k_means_clustering(fashion_mnist, max_clusters=N_Clusters,
                                                                  folder=folder, dataset_name=f"Fashion_MNIST_{extra}")
                run_expectation_maximization(fashion_mnist, folder=folder, dataset_name=f"Fashion_MNIST_{extra}")
        return
    except Exception as run_clustering_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'run_clustering'", run_clustering_exception)


def run_dimensionality_reduction(mnist, fashion_mnist, run_check=None, scaled=False):
    try:
        folder = "/DimensionalityReduction/"
        utl.check_folder(folder)
        if run_check is not None:
            if run_check["Run PCA"]:
                run_pca(dataset=mnist, folder=folder, dataset_name=f"MNIST_{extra}")
            if run_check["Run ICA"]:
                run_ica(dataset=mnist, folder=folder, dataset_name=f"MNIST_{extra}")
            if run_check["Run Randomized Projections"]:
                run_random_projections(dataset=mnist, folder=folder, dataset_name=f"MNIST_{extra}")
            if run_check["Run TSNE"]:
                run_tsne(dataset=mnist, folder=folder, dataset_name=f"MNIST_{extra}")
            if not TESTING:
                if run_check["Run PCA"]:
                    run_pca(dataset=fashion_mnist, folder=folder, dataset_name=f"Fashion_MNIST_{extra}")
                if run_check["Run ICA"]:
                    run_ica(dataset=fashion_mnist, folder=folder, dataset_name=f"Fashion_MNIST_{extra}")
                if run_check["Run Randomized Projections"]:
                    run_random_projections(dataset=fashion_mnist, folder=folder, dataset_name=f"Fashion_MNIST_{extra}")
                if run_check["Run TSNE"]:
                    run_tsne(dataset=fashion_mnist, folder=folder, dataset_name=f"Fashion_MNIST_{extra}")
        else:
            run_pca(dataset=mnist, folder=folder, dataset_name=f"MNIST_{extra}")
            run_ica(dataset=mnist, folder=folder, dataset_name=f"MNIST_{extra}")
            run_random_projections(dataset=mnist, folder=folder, dataset_name=f"MNIST_{extra}")
            run_tsne(dataset=mnist, folder=folder, dataset_name=f"MNIST_{extra}")
            if not TESTING:
                run_pca(dataset=fashion_mnist, folder=folder, dataset_name=f"Fashion_MNIST_{extra}")
                run_ica(dataset=fashion_mnist, folder=folder, dataset_name=f"Fashion_MNIST_{extra}")
                run_random_projections(dataset=fashion_mnist, folder=folder, dataset_name=f"Fashion_MNIST_{extra}")
                run_tsne(dataset=fashion_mnist, folder=folder, dataset_name=f"Fashion_MNIST_{extra}")
        return
    except Exception as run_dimensionality_reduction_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'run_dimensionality_reduction'", run_dimensionality_reduction_exception)


def start(scale=False):
    try:
        if TESTING:
            gathered_data = utl.setup(["MNIST"])
            mnist = {}
            mnist['train_X'], mnist['train_y'], \
            mnist['valid_X'], mnist['valid_y'], \
            mnist['test_X'], mnist['test_y'] = utl.split_data(gathered_data["MNIST"]["X"],
                                                              gathered_data["MNIST"]["y"], scale=scale)

            return mnist
        else:
            gathered_data = utl.setup(["MNIST"])
            gathered_data_fashion = utl.setup(["Fashion-MNIST"])
            mnist = {}
            fashion_mnist = {}
            mnist['train_X'], mnist['train_y'], \
            mnist['valid_X'], mnist['valid_y'], \
            mnist['test_X'], mnist['test_y'] = utl.split_data(gathered_data["MNIST"]["X"],
                                                              gathered_data["MNIST"]["y"], scale=scale)

            fashion_mnist['train_X'], fashion_mnist['train_y'], \
            fashion_mnist['valid_X'], fashion_mnist['valid_y'], \
            fashion_mnist['test_X'], fashion_mnist['test_y'] = utl.split_data(
                gathered_data_fashion["Fashion-MNIST"]["X"],
                gathered_data_fashion["Fashion-MNIST"]["y"], scale=scale)

            return mnist, fashion_mnist

    except Exception as start_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'start'", start_exception)


def run_part_1():
    try:
        # Using just the MNIST dataset, compare preprocessing results with both K-Means and EM
        gathered_data = utl.setup(["MNIST"])
        mnist_scaled = {}
        mnist_scaled['train_X'], mnist_scaled['train_y'], \
        mnist_scaled['valid_X'], mnist_scaled['valid_y'], \
        mnist_scaled['test_X'], mnist_scaled['test_y'] = utl.split_data(gathered_data["MNIST"]["X"],
                                                                        gathered_data["MNIST"]["y"], scale=True)
        mnist_not_scaled = {}
        mnist_not_scaled['train_X'], mnist_not_scaled['train_y'], \
        mnist_not_scaled['valid_X'], mnist_not_scaled['valid_y'], \
        mnist_not_scaled['test_X'], mnist_not_scaled['test_y'] = utl.split_data(gathered_data["MNIST"]["X"],
                                                                                gathered_data["MNIST"]["y"],
                                                                                scale=False)


        return
    except Exception as run_part_1_exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(f"Exception in 'run_part_1'", run_part_1_exception)


if __name__ == "__main__":
    cluster_run = {"Run K-Means Clustering": True, "Run Expectation Maximization": False}
    dimension_run = {"Run PCA": False, "Run ICA": False, "Run Randomized Projections": False,
                     "Run TSNE": False}
    scale_data = True
    if TESTING:
        mnist_data = start(scale_data)
        run_clustering(mnist_data, None, run_check=cluster_run, scaled=scale_data)
        run_dimensionality_reduction(mnist_data, None, run_check=dimension_run, scaled=scale_data)
    else:
        mnist_data, fashion_mnist_data = start(scale_data)
        run_clustering(mnist_data, fashion_mnist_data, run_check=cluster_run, scaled=scale_data)
        run_dimensionality_reduction(mnist_data, fashion_mnist_data, run_check=dimension_run, scaled=scale_data)
    print()
