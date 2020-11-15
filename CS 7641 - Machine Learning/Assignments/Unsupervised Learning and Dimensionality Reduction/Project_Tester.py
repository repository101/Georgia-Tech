import os
import pickle
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import clustering_utl as cl_utl
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

import unsupervised_learning_util as utl

plt.tight_layout()
plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500

NJOBS = 32
VERBOSE = 0
limit = 10000

folder = "DimensionalityReduction/"
utl.check_folder(folder)

if __name__ == "__main__":
	gathered_data = utl.setup(["MNIST"])
	gathered_data_fashion = utl.setup(["Fashion-MNIST"])

	mnist = {}
	fashion_mnist = {}
	mnist_not_scaled = {}
	fashion_mnist_not_scaled = {}

	mnist['train_X'], mnist['train_y'], \
	mnist['valid_X'], mnist['valid_y'], \
	mnist['test_X'], mnist['test_y'] = utl.split_data(gathered_data["MNIST"]["X"],
	                                                  gathered_data["MNIST"]["y"], minMax=True)
	mnist_not_scaled['train_X'], mnist_not_scaled['train_y'], \
	mnist_not_scaled['valid_X'], mnist_not_scaled['valid_y'], \
	mnist_not_scaled['test_X'], mnist_not_scaled['test_y'] = utl.split_data(
		gathered_data["MNIST"]["X"], gathered_data["MNIST"]["y"], scale=False)

	fashion_mnist['train_X'], fashion_mnist['train_y'], \
	fashion_mnist['valid_X'], fashion_mnist['valid_y'], \
	fashion_mnist['test_X'], fashion_mnist['test_y'] = utl.split_data(gathered_data_fashion["Fashion-MNIST"]["X"],
	                                                                  gathered_data_fashion["Fashion-MNIST"]["y"],
	                                                                  minMax=True)

	fashion_mnist_not_scaled['train_X'], fashion_mnist_not_scaled['train_y'], \
	fashion_mnist_not_scaled['valid_X'], fashion_mnist_not_scaled['valid_y'], \
	fashion_mnist_not_scaled['test_X'], fashion_mnist_not_scaled['test_y'] = utl.split_data(
		gathered_data_fashion["Fashion-MNIST"]["X"], gathered_data_fashion["Fashion-MNIST"]["y"], scale=False)

	# results = cl_utl.run_kmeans(data_X=mnist["train_X"].loc[:5000, :], data_y=mnist["train_y"].loc[:5000],
	#                             max_clusters=20, dataset_name="MNIST", verbose=1)

	# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
	# em_mnist_results = cl_utl.run_em(data_X=mnist["train_X"].iloc[:limit, :], data_y=mnist["train_y"].iloc[:limit],
	#                                  max_components=30, dataset_name="MNIST", standalone=False, ax=ax1)
	# em_fashion_results = cl_utl.run_em(data_X=fashion_mnist["train_X"].iloc[:limit, :], data_y=fashion_mnist["train_y"].iloc[:limit],
	#                                  max_components=30, dataset_name="Fashion-MNIST", standalone=False, ax=ax2)
	# plt.savefig(f"{os.getcwd()}/Clustering/EM_Combined_AIC_BIC.png")
	# with open(f"{os.getcwd()}/DimensionalityReduction/PCA_Fashion_Reduced_Dataset.pkl", "rb") as input_file:
	# 	pca_fashion_reduced_data = pickle.load(input_file)
	# 	input_file.close()
	# print(pca_fashion_reduced_data.shape)
	#
	# with open(f"{os.getcwd()}/DimensionalityReduction/ICA_Fashion_Reduced_Dataset.pkl", "rb") as input_file:
	# 	ica_fashion_reduced_data = pickle.load(input_file)
	# 	input_file.close()
	# print(ica_fashion_reduced_data.shape)
	#
	# with open(f"{os.getcwd()}/DimensionalityReduction/RP_Fashion_Reduced_Dataset.pkl", "rb") as input_file:
	# 	rp_fashion_reduced_data = pickle.load(input_file)
	# 	input_file.close()
	# print(rp_fashion_reduced_data.shape)
	#
	# with open(f"{os.getcwd()}/DimensionalityReduction/RF_Fashion_Reduced_Dataset.pkl", "rb") as input_file:
	# 	rf_fashion_reduced_data = pickle.load(input_file)
	# 	input_file.close()
	# print(rf_fashion_reduced_data.shape)

	# with open(f"{os.getcwd()}/Clustering/KMeans_Clustered_Reduced_Dataset.pkl", "rb") as input_file:
	# 	kmeans_reduced_data = pickle.load(input_file)
	# 	input_file.close()
	#
	# with open(f"{os.getcwd()}/Clustering/Expecation_Maximization_Clustered_Reduced_Dataset.pkl", "rb") as input_file:
	# 	em_reduced_data = pickle.load(input_file)
	# 	input_file.close()

	# t = kmeans_reduced_data.max(axis=1)
	# p = kmeans_reduced_data.max(axis=0)
	# d = kmeans_reduced_data.max()
	# cl_utl.run_cluster_nn(kmeans_X=kmeans_reduced_data.iloc[:, :-1], kmeans_y=kmeans_reduced_data.iloc[:, -1],
	#                       expectation_X=em_reduced_data.iloc[:, :-1], expectation_y=em_reduced_data.iloc[:, -1],
	#                       base_line_data_X=fashion_mnist['train_X'].iloc[:10000, :],
	#                       base_line_data_y=fashion_mnist['train_y'].iloc[:10000])

	# cl_utl.run_nn(pca_data_X=pca_fashion_reduced_data.iloc[:, :-1], pca_data_y=pca_fashion_reduced_data.iloc[:, -1],
	#               ica_data_X=ica_fashion_reduced_data.iloc[:, :-1], ica_data_y=ica_fashion_reduced_data.iloc[:, -1],
	#               rand_proj_data_X=rp_fashion_reduced_data.iloc[:, :-1],
	#               rand_proj_data_y=rp_fashion_reduced_data.iloc[:, -1],
	#               rand_forest_data_X=rf_fashion_reduced_data.iloc[:, :-1],
	#               rand_forest_data_y=rf_fashion_reduced_data.iloc[:, -1],
	#               base_line_data_X=fashion_mnist["train_X"].iloc[:limit, :],
	#               base_line_data_y=fashion_mnist["train_y"].iloc[:limit])
	with open(f"{os.getcwd()}/DimensionalityReduction/PrincipleComponentAnalysis/"
	          f"MNIST_PCA_Results.pkl", "rb") as input_file:
		pca_mnist_results = pickle.load(input_file)
		input_file.close()

	with open(f"{os.getcwd()}/DimensionalityReduction/PrincipleComponentAnalysis/"
	          f"Fashion_PCA_Results.pkl", "rb") as input_file:
		pca_fashion_results = pickle.load(input_file)
		input_file.close()
	with open(f"{os.getcwd()}/DimensionalityReduction/"
	          f"IndependentComponentAnalysis/MNIST_ICA_Results.pkl", "rb") as input_file:
		ica_results_mnist = pickle.load(input_file)
		input_file.close()
	with open(f"{os.getcwd()}/DimensionalityReduction/"
	          f"IndependentComponentAnalysis/Fashion_MNIST_ICA_Results.pkl", "rb") as input_file:
		ica_results_fashion = pickle.load(input_file)
		input_file.close()
	with open(f"{os.getcwd()}/DimensionalityReduction/"
	          f"RandomProjections/MNIST_RP_results.pkl", "rb") as input_file:
		rp_results_mnist = pickle.load(input_file)
		input_file.close()

	with open(f"{os.getcwd()}/DimensionalityReduction/"
	          f"RandomProjections/Fashion_RP_results.pkl", "rb") as input_file:
		rp_results_fashion = pickle.load(input_file)
		input_file.close()

	with open(f"{os.getcwd()}/DimensionalityReduction/RF_MNIST_Reduced_Dataset.pkl", "rb") as input_file:
		rf_mnist_reduced_data = pickle.load(input_file)
		input_file.close()
	with open(f"{os.getcwd()}/DimensionalityReduction/RF_Fashion_Reduced_Dataset.pkl", "rb") as input_file:
		rf_fashion_reduced_data = pickle.load(input_file)
		input_file.close()
	print(f"RF MNIST: {rf_mnist_reduced_data.shape}")
	print(f"RF Fashion: {rf_fashion_reduced_data.shape}")
	with open(f"{os.getcwd()}/DimensionalityReduction/RP_MNIST_Reduced_Dataset.pkl", "rb") as input_file:
		rp_mnist_reduced_data = pickle.load(input_file)
		input_file.close()
	with open(f"{os.getcwd()}/DimensionalityReduction/RP_Fashion_Reduced_Dataset.pkl", "rb") as input_file:
		rp_fashion_reduced_data = pickle.load(input_file)
		input_file.close()
	print(f"RP MNIST: {rp_mnist_reduced_data.shape}")
	print(f"RP Fashion: {rp_fashion_reduced_data.shape}")
	with open(f"{os.getcwd()}/DimensionalityReduction/ICA_MNIST_Reduced_Dataset.pkl", "rb") as input_file:
		ica_mnist_reduced_data = pickle.load(input_file)
		input_file.close()
	with open(f"{os.getcwd()}/DimensionalityReduction/ICA_Fashion_Reduced_Dataset.pkl", "rb") as input_file:
		ica_fashion_reduced_data = pickle.load(input_file)
		input_file.close()
	print(f"ICA MNIST: {ica_mnist_reduced_data.shape}")
	print(f"ICA Fashion: {ica_fashion_reduced_data.shape}")
	with open(f"{os.getcwd()}/DimensionalityReduction/PCA_MNIST_Reduced_Dataset.pkl", "rb") as input_file:
		pca_mnist_reduced_data = pickle.load(input_file)
		input_file.close()
	with open(f"{os.getcwd()}/DimensionalityReduction/PCA_Fashion_Reduced_Dataset.pkl", "rb") as input_file:
		pca_fashion_reduced_data = pickle.load(input_file)
		input_file.close()
	print(f"PCA MNIST: {pca_mnist_reduced_data.shape}")
	print(f"PCA Fashion: {pca_fashion_reduced_data.shape}")
	with open(f"{os.getcwd()}/DimensionalityReduction/RandomForest/RandomForest_MNIST_Results.pkl", "rb") as input_file:
		rf_results_mnist = pickle.load(input_file)
		input_file.close()
	with open(f"{os.getcwd()}/DimensionalityReduction/RandomForest/RandomForest_Fashion_Results.pkl",
	          "rb") as input_file:
		rf_results_fashion = pickle.load(input_file)
		input_file.close()

	lim = 10000
	# RP

	utl.plot_better_results(data_X=rp_mnist_reduced_data.iloc[:lim, :-1], data_y=rp_mnist_reduced_data.iloc[:lim, -1],
	                        data_results=rp_results_mnist, dataset_name="MNIST",
	                        algorithm_name="Randomized_Projections",
	                        vline_idx=715, pixel_importance=None, model=None,
	                        original_data_X=mnist["train_X"].iloc[:lim, :],
	                        original_data_y=mnist["train_y"].iloc[:lim],
	                        results=rp_results_mnist, is_rand_proj=True, is_fashion=False, font_size=14, n_clusters=12)

	utl.plot_better_results(data_X=rp_fashion_reduced_data.iloc[:lim, :-1], data_y=rp_fashion_reduced_data.iloc[:lim, -1],
	                        data_results=rp_results_fashion, dataset_name="Fashion_MNIST",
	                        algorithm_name="Randomized_Projections",
	                        vline_idx=746, pixel_importance=None, model=None,
	                        original_data_X=fashion_mnist["train_X"].iloc[:lim, :],
	                        original_data_y=fashion_mnist["train_y"].iloc[:lim],
	                        results=rp_results_fashion, is_rand_proj=True, is_fashion=True, font_size=14, n_clusters=10)

	# ICA
	temp_ica_mnist_results = FastICA(n_components=284, whiten=True, max_iter=400).fit(mnist["train_X"].iloc[:lim, :])
	print("Finished training MNIST ICA")
	utl.plot_better_results(data_X=ica_mnist_reduced_data.iloc[:lim, :-1], data_y=ica_mnist_reduced_data.iloc[:lim, -1],
	                        data_results=ica_results_mnist, dataset_name="MNIST", algorithm_name="ICA",
	                        vline_idx=284, pixel_importance=temp_ica_mnist_results.mean_, model=temp_ica_mnist_results,
	                        original_data_X=mnist["train_X"].iloc[:lim, :],
	                        original_data_y=mnist["train_y"].iloc[:lim],
	                        results=ica_results_mnist, is_ica=True, is_fashion=False, font_size=14, n_clusters=35)

	temp_ica_fashion_results = FastICA(n_components=412, whiten=True,
	                                   max_iter=400).fit(fashion_mnist["train_X"].iloc[:lim, :])
	print("Finished training Fashion ICA")
	utl.plot_better_results(data_X=ica_fashion_reduced_data.iloc[:lim, :-1],
	                        data_y=ica_fashion_reduced_data.iloc[:lim, -1],
	                        data_results=ica_results_fashion, dataset_name="Fashion_MNIST", algorithm_name="ICA",
	                        vline_idx=412, pixel_importance=temp_ica_fashion_results.mean_,
	                        model=temp_ica_fashion_results,
	                        original_data_X=fashion_mnist["train_X"].iloc[:lim, :],
	                        original_data_y=fashion_mnist["train_y"].iloc[:lim],
	                        results=ica_results_fashion, is_ica=True, is_fashion=True, font_size=14, n_clusters=15)

	PCA
	temp_pca_mnist_results = PCA(whiten=True, svd_solver="full").fit(mnist["train_X"].iloc[:lim, :])
	mnist_component_df = pd.DataFrame(temp_pca_mnist_results.components_)
	mnist_pixel_importance = mnist_component_df.idxmax(axis=1)
	utl.plot_better_results(data_X=pca_mnist_reduced_data.iloc[:lim, :-1], data_y=pca_mnist_reduced_data.iloc[:lim, -1],
	                        data_results=pca_mnist_results, dataset_name="MNIST", algorithm_name="PCA",
	                        vline_idx=326, pixel_importance=mnist_pixel_importance.values, model=temp_pca_mnist_results,
	                        original_data_X=mnist["train_X"].iloc[:lim, :],
	                        original_data_y=mnist["train_y"].iloc[:lim],
	                        results=pca_mnist_results, is_pca=True, is_fashion=False, font_size=14, n_clusters=13)

	temp_pca_fashion_results = PCA(whiten=True, svd_solver="full").fit(fashion_mnist["train_X"].iloc[:lim, :])
	fashion_component_df = pd.DataFrame(temp_pca_fashion_results.components_)
	fashion_pixel_importance = fashion_component_df.idxmax(axis=1)
	utl.plot_better_results(data_X=pca_fashion_reduced_data.iloc[:lim, :-1],
	                        data_y=pca_fashion_reduced_data.iloc[:lim, -1],
	                        data_results=pca_fashion_results, dataset_name="Fashion_MNIST", algorithm_name="PCA",
	                        vline_idx=445, pixel_importance=fashion_pixel_importance.values,
	                        model=temp_pca_fashion_results,
	                        original_data_X=fashion_mnist["train_X"].iloc[:lim, :],
	                        original_data_y=fashion_mnist["train_y"].iloc[:lim],
	                        results=pca_fashion_results, is_pca=True, is_fashion=True, font_size=14, n_clusters=11)


	RF
	utl.plot_better_results(data_X=rf_mnist_reduced_data.iloc[:lim, :-1],
	                        data_y=rf_mnist_reduced_data.iloc[:lim, -1],
	                        data_results=rf_results_mnist, dataset_name="MNIST", algorithm_name="Random Forest",
	                        vline_idx=152, pixel_importance=None,
	                        model=None,
	                        original_data_X=mnist["train_X"].iloc[:lim, :],
	                        original_data_y=mnist["train_y"].iloc[:lim],
	                        results=rf_results_mnist, is_rand_forest=True, is_fashion=False, font_size=14,
	                        n_clusters=11)

	utl.plot_better_results(data_X=rf_fashion_reduced_data.iloc[:lim, :-1],
	                        data_y=rf_fashion_reduced_data.iloc[:lim, -1],
	                        data_results=rf_results_fashion, dataset_name="Fashion_MNIST", algorithm_name="Random Forest",
	                        vline_idx=250, pixel_importance=None,
	                        model=None,
	                        original_data_X=fashion_mnist["train_X"].iloc[:lim, :],
	                        original_data_y=fashion_mnist["train_y"].iloc[:lim],
	                        results=rf_results_fashion, is_rand_forest=True, is_fashion=True, font_size=14,
	                        n_clusters=10)
	print()
