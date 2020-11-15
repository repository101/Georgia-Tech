import glob
import os
import pickle
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import joblib
from joblib import delayed, Parallel, parallel_backend
import pandas as pd
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

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500


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