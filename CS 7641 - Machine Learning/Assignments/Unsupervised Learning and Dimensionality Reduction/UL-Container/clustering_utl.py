import os
import os
import pickle
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score, homogeneity_completeness_v_measure, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from yellowbrick.model_selection import LearningCurve
from yellowbrick.utils.kneed import KneeLocator

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500


def run_kmeans(data_X, data_y, max_clusters, dataset_name, verbose=1, save_name=""):
	try:
		cols = ["Inertia", "Silhouette", "Homogeneity", "Completeness", "Harmonic_Mean", "Calinski_Harabasz",
		        "Davies_Bouldin"]
		idx = [i for i in range(2, max_clusters, 1)]
		result_df = pd.DataFrame(columns=cols, index=idx, data=np.zeros(shape=(len(idx), len(cols))))

		print("Starting K-Means Clustering")
		for k in idx:
			k_means = KMeans(n_clusters=k, verbose=verbose).fit(data_X)
			inertia = k_means.inertia_
			# silhouette_average = silhouette_score(data_X, k_means.labels_, sample_size=limit)
			silhouette_average = silhouette_score(data_X, k_means.labels_)
			homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(data_y, k_means.labels_)
			result_df.loc[k, "Inertia"] = inertia
			result_df.loc[k, "Silhouette"] = silhouette_average
			result_df.loc[k, "Calinski_Harabasz"] = calinski_harabasz_score(data_X, k_means.labels_)
			result_df.loc[k, "Davies_Bouldin"] = davies_bouldin_score(data_X, k_means.labels_)
			result_df.loc[k, "Homogeneity"] = homogeneity
			result_df.loc[k, "Completeness"] = completeness
			result_df.loc[k, "Harmonic_Mean"] = v_measure
			print(f"\n\t{dataset_name} - k={k} \n{result_df.loc[k]}")

		model = KMeans()
		plt.close("all")
		fig, ax1 = plt.subplots()
		visualizer = KElbowVisualizer(model, k=(2, max_clusters), ax=ax1, timings=False)
		visualizer.fit(data_X).finalize()
		elbow = visualizer.elbow_value_

		ax1.set_title(f"K Means Clustering\nDistortion MNIST", fontsize=15, weight='bold')
		ax1.set_xlabel("K Clusters", fontsize=15, weight='heavy')
		ax1.set_ylabel("Distortion", fontsize=15, weight='heavy')
		ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True, shadow=True)

		plt.savefig(f"{os.getcwd()}/Clustering/KMeans_Elbow_{dataset_name}_{save_name}.png", bbox_inches='tight')
		plt.close("all")
		fig, ax1 = plt.subplots()
		cluster_count = elbow

		model = KMeans(n_clusters=elbow, random_state=42)

		solhouette_vis = SilhouetteVisualizer(model, ax=ax1, colors='yellowbrick').fit(data_X).finalize()

		ax1.set_title(f"Silhouette Plot of KMeans Clustering\non MNIST with {cluster_count} Clusters",
		              fontsize=15, weight='bold')
		ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True, shadow=True)
		ax1.set_xlabel("Silhouette Coefficient Values", fontsize=15, weight='heavy')
		ax1.set_ylabel("Cluster Label", fontsize=15, weight='heavy')

		plt.tight_layout()

		plt.savefig(f"{os.getcwd()}/Clustering/KMeans_Silhouette_{dataset_name}_{save_name}.png", bbox_inches='tight')
		results = {"Result_DF": result_df, "Elbow": elbow}
		return results
	except Exception as run_kmeans_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'run_kmeans'", run_kmeans_exception)


def plot_combined_kmeans(mnist_X, fashion_X, max_clusters, save_name="Combined"):
	try:
		plt.close("all")
		fig, ((ax1_1, ax1_2), (ax2_1, ax2_2)) = plt.subplots(2, 2, figsize=(12, 12))
		k_mnist_model = KMeans()
		print("Starting K-Means Elbow Visualizer for MNIST")
		mnist_visualizer = KElbowVisualizer(k_mnist_model, k=(2, max_clusters), ax=ax1_1, timings=False)
		mnist_visualizer.fit(mnist_X).finalize()
		print("Finished K-Means Elbow Visualizer for MNIST")
		mnist_elbow = mnist_visualizer.elbow_value_

		ax1_1.set_title(f"K Means Clustering\nDistortion MNIST", fontsize=15, weight='bold')
		ax1_1.set_xlabel("K Clusters", fontsize=15, weight='heavy')
		ax1_1.set_ylabel("Distortion", fontsize=15, weight='heavy')
		# ax1_1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True, shadow=True)

		mnist_model = KMeans(n_clusters=mnist_elbow, random_state=42)
		print("Starting K-Means Silhouette Visualizer for MNIST")
		mnist_solhouette_vis = SilhouetteVisualizer(mnist_model, ax=ax1_2, colors='yellowbrick').fit(mnist_X).finalize()
		print("Finished K-Means Silhouette Visualizer for MNIST")
		ax1_2.set_title(f"Silhouette Plot of KMeans Clustering\non MNIST with {mnist_elbow} Clusters",
		                fontsize=15, weight='bold')
		# ax1_2.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True, shadow=True)
		ax1_2.set_xlabel("Silhouette Coefficient Values", fontsize=15, weight='heavy')
		ax1_2.set_ylabel("Cluster Label", fontsize=15, weight='heavy')
		print("Finished MNIST Section")

		k_fashion_model = KMeans()
		print("Starting K-Means Elbow Visualizer for Fashion-MNIST")
		fashion_visualizer = KElbowVisualizer(k_fashion_model, k=(2, max_clusters), ax=ax2_1, timings=False)
		fashion_visualizer.fit(fashion_X).finalize()
		print("Finished K-Means Elbow Visualizer for Fashion-MNIST")
		fashion_elbow = fashion_visualizer.elbow_value_

		ax2_1.set_title(f"K Means Clustering\nDistortion Fashion MNIST", fontsize=15, weight='bold')
		ax2_1.set_xlabel("K Clusters", fontsize=15, weight='heavy')
		ax2_1.set_ylabel("Distortion", fontsize=15, weight='heavy')
		# ax2_1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True, shadow=True)

		fashion_model = KMeans(n_clusters=fashion_elbow, random_state=42)
		print("Starting K-Means Silhouette Visualizer for Fashion-MNIST")
		fashion_solhouette_vis = SilhouetteVisualizer(fashion_model,
		                                              ax=ax2_2, colors='yellowbrick').fit(fashion_X).finalize()
		print("Finished K-Means Silhouette Visualizer for Fashion-MNIST")
		ax2_2.set_title(f"Silhouette Plot of KMeans Clustering\non Fashion-MNIST with {fashion_elbow} Clusters",
		                fontsize=15, weight='bold')
		# ax1_2.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True, shadow=True)
		ax2_2.set_xlabel("Silhouette Coefficient Values", fontsize=15, weight='heavy')
		ax2_2.set_ylabel("Cluster Label", fontsize=15, weight='heavy')
		print("Finished Fashion-MNIST")
		plt.tight_layout()
		plt.savefig(f"{os.getcwd()}/Clustering/Distortion_Silhouette_{save_name}_Combined.png")
		return
	except Exception as plot_combined_kmeans_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'plot_combined_kmeans'", plot_combined_kmeans_exception)


def run_em(data_X, data_y, max_components, dataset_name, ax=None, standalone=True):
	try:
		def plot_samples(X, Y, n_components, index, title):
			# https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_sin.html#sphx-glr-auto-examples-mixture-plot-gmm-sin-py
			plt.subplot(10, 5, 4 + index)
			for i in range(n_components):
				# as the DP will not use every component it has access to
				# unless it needs it, we shouldn't plot the redundant
				# components.
				if not np.any(Y == i):
					continue
				plt.scatter(X[Y == i, 0], X[Y == i, 1], .8)

			plt.title(title)
			plt.xticks(())
			plt.yticks(())

		index = np.arange(2, max_components, 1).astype(np.int)
		types = ["Full"]
		columns = ["AIC_Full", "BIC_Full"]

		results = pd.DataFrame(columns=columns, index=index, data=np.zeros(shape=(index.shape[0], len(columns))))
		for idx in index:
			print(f"N_Components: {idx}")
			for _type in types:
				temp_gmm = GaussianMixture(n_components=idx, n_init=10, covariance_type=_type.lower(),
				                           warm_start=True, max_iter=500).fit(data_X)
				results.loc[idx, f"AIC_{_type}"] = temp_gmm.aic(data_X)
				results.loc[idx, f"BIC_{_type}"] = temp_gmm.bic(data_X)

		with open(f"{os.getcwd()}/Clustering/EM_Results_{dataset_name}.pkl", "wb") as output_file:
			pickle.dump(results, output_file)
			output_file.close()

		if standalone:
			plt.close("all")
			fig, ax1 = plt.subplots()

			results[["AIC_Full", "BIC_Full"]].plot(ax=ax1)
			ax1.set_title(f"AIC / BIC Comparison\n {dataset_name}", fontsize=15, weight='bold')
			ax1.legend(loc="best", markerscale=1.1, frameon=True,
			           edgecolor="black", fancybox=True, shadow=True)
			ax1.set_xlabel("N Components", fontsize=15, weight='heavy')
			ax1.set_ylabel("Information Criterion", fontsize=15, weight='heavy')

			plt.tight_layout()

			plt.savefig(f"{os.getcwd()}/Clustering/EM_AicBic_{dataset_name}.png", bbox_inches='tight')
			return results
		else:
			results[["AIC_Full", "BIC_Full"]].plot(ax=ax)
			ax.set_title(f"AIC / BIC Comparison\n {dataset_name}", fontsize=15, weight='bold')
			ax.legend(loc="best", markerscale=1.1, frameon=True,
			          edgecolor="black", fancybox=True, shadow=True)
			ax.set_xlabel("N Components", fontsize=15, weight='heavy')
			ax.set_ylabel("Information Criterion", fontsize=15, weight='heavy')

			plt.tight_layout()
			return results
	except Exception as run_em_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'run_em'", run_em_exception)


def find_elbow(data_X, data_y, direction='decreasing'):
	try:
		knee_obj = KneeLocator(x=data_X, y=data_y, curve_direction=direction)
		knee_obj.find_knee()
		return knee_obj.elbow
	except Exception as find_elbow_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'find_elbow'", find_elbow_exception)


def plot_em(mnist_results, fashion_results, extra_name):
	try:
		plt.close("all")
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

		ax1.plot(mnist_results["AIC_Full"], label="AIC")
		ax1.plot(mnist_results["BIC_Full"], label="BIC")

		ax1.set_title(f"AIC / BIC Comparison\n MNIST", fontsize=15, weight='bold')
		ax1.legend(loc="best", markerscale=1.1, frameon=True,
		           edgecolor="black", fancybox=True, shadow=True)
		ax1.set_xlabel("N Components", fontsize=15, weight='heavy')
		ax1.set_ylabel("Information Criterion", fontsize=15, weight='heavy')

		ax2.plot(fashion_results["AIC_Full"], label="AIC")
		ax2.plot(fashion_results["BIC_Full"], label="BIC")

		ax2.set_title(f"AIC / BIC Comparison\n Fashion-MNIST", fontsize=15, weight='bold')
		ax2.legend(loc="best", markerscale=1.1, frameon=True,
		           edgecolor="black", fancybox=True, shadow=True)
		ax2.set_xlabel("N Components", fontsize=15, weight='heavy')
		ax2.set_ylabel("Information Criterion", fontsize=15, weight='heavy')
		plt.tight_layout()
		plt.savefig(f"{os.getcwd()}/Clustering/EM_AIC_BIC_Combined_Final_{extra_name}.png")
		return
	except Exception as plot_em_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'plot_em'", plot_em_exception)


def run_nn(pca_data_X, pca_data_y, ica_data_X, ica_data_y, rand_proj_data_X, rand_proj_data_y, rand_forest_data_X,
           rand_forest_data_y, base_line_data_X, base_line_data_y, dataset_name="Fashion-MNIST", max_iters=600,
           alpha=0.1):
	try:
		rf_num_columns = rand_forest_data_X.shape[1]
		rand_forest_data_X.columns = [i for i in range(rf_num_columns)]

		cv = 10
		n_jobs = 32
		sizes = np.round(np.arange(0.05, 1.01, 0.05), 2)
		plot_sizes = sizes * 8000
		cols = ["Baseline", "PCA", "ICA", "Random_Projection", "Random_Forest"]
		train_results = pd.DataFrame(index=sizes, columns=cols)
		test_results = pd.DataFrame(index=sizes, columns=cols)
		train_times = pd.DataFrame(index=sizes, columns=cols)
		predict_times = pd.DataFrame(index=sizes, columns=cols)

		plt.close("all")
		fig, ax = plt.subplots()
		# print(f"\n\tStarting Baseline")
		# base_model = MLPClassifier(hidden_layer_sizes=(40,), solver='adam', max_iter=max_iters, alpha=1.0,
		#                            verbose=True, random_state=42, warm_start=True)
		# base_visualizer = LearningCurve(base_model, cv=cv, scoring='accuracy', train_sizes=sizes, n_jobs=n_jobs,
		#                                 ax=ax)
		# base_visualizer.fit(base_line_data_X, base_line_data_y)
		# train_results["Baseline"] = base_visualizer.train_scores_mean_
		# test_results["Baseline"] = base_visualizer.test_scores_mean_
		# train_times["Baseline"] = base_visualizer.fit_times_mean_
		# predict_times["Baseline"] = base_visualizer.predict_times_mean_
		# print(f"\n\tFinished Baseline")

		# PCA
		print(f"\n\tStarting Principal Component Analysis")
		pca_nn_clf = MLPClassifier(hidden_layer_sizes=(40,), solver='adam', max_iter=max_iters, alpha=alpha,
		                           verbose=True, random_state=42)
		pca_visualizer = LearningCurve(pca_nn_clf, cv=cv, scoring='accuracy', train_sizes=sizes,
		                               n_jobs=n_jobs, ax=ax)
		pca_visualizer.fit(pca_data_X, pca_data_y)
		train_results["PCA"] = pca_visualizer.train_scores_mean_
		test_results["PCA"] = pca_visualizer.test_scores_mean_
		train_times["PCA"] = pca_visualizer.fit_times_mean_
		predict_times["PCA"] = pca_visualizer.predict_times_mean_
		print(f"\n\tFinished Principal Component Analysis")

		# ICA
		print(f"\n\tStarting Independent Component Analysis")
		ica_nn_clf = MLPClassifier(hidden_layer_sizes=(40,), solver='adam', max_iter=max_iters, alpha=alpha,
		                           verbose=True, random_state=42)
		ica_visualizer = LearningCurve(ica_nn_clf, cv=cv, scoring='accuracy', train_sizes=sizes,
		                               n_jobs=n_jobs, ax=ax)
		ica_visualizer.fit(ica_data_X, ica_data_y)
		train_results["ICA"] = ica_visualizer.train_scores_mean_
		test_results["ICA"] = ica_visualizer.test_scores_mean_
		train_times["ICA"] = ica_visualizer.fit_times_mean_
		predict_times["ICA"] = ica_visualizer.predict_times_mean_
		print(f"\n\tFinished Independent Component Analysis")

		# Randomized Projection
		print(f"\n\tStarting Randomized Projections")
		rand_proj_nn_clf = MLPClassifier(hidden_layer_sizes=(40,), solver='adam', max_iter=max_iters, alpha=alpha,
		                                 verbose=True, random_state=42)
		rand_proj_visualizer = LearningCurve(rand_proj_nn_clf, cv=cv, scoring='accuracy', train_sizes=sizes,
		                                     n_jobs=n_jobs, ax=ax)
		rand_proj_visualizer.fit(rand_proj_data_X, rand_proj_data_y)
		train_results["Random_Projection"] = rand_proj_visualizer.train_scores_mean_
		test_results["Random_Projection"] = rand_proj_visualizer.test_scores_mean_
		train_times["Random_Projection"] = rand_proj_visualizer.fit_times_mean_
		predict_times["Random_Projection"] = rand_proj_visualizer.predict_times_mean_
		print(f"\n\tFinished Randomized Projections")

		# Random Forest
		print(f"\n\tStarting Random Forest")
		rand_forest_nn_clf = MLPClassifier(hidden_layer_sizes=(40,), solver='adam', max_iter=max_iters, alpha=alpha,
		                                   verbose=True, random_state=42)
		rand_forest_visualizer = LearningCurve(rand_forest_nn_clf, cv=cv, scoring='accuracy', train_sizes=sizes,
		                                       n_jobs=n_jobs, ax=ax)
		rand_forest_visualizer.fit(rand_forest_data_X, rand_forest_data_y)
		train_results["Random_Forest"] = rand_forest_visualizer.train_scores_mean_
		test_results["Random_Forest"] = rand_forest_visualizer.test_scores_mean_
		train_times["Random_Forest"] = rand_forest_visualizer.fit_times_mean_
		predict_times["Random_Forest"] = rand_forest_visualizer.predict_times_mean_
		print(f"\n\tFinished Random Forest")

		plt.close("all")
		mpl.style.use("ggplot")
		fig, ((pca_ax, ica_ax), (rp_ax, rf_ax)) = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
		pca_ax_secondary = pca_ax.twinx()
		ica_ax_secondary = ica_ax.twinx()
		rp_ax_secondary = rp_ax.twinx()
		rf_ax_secondary = rf_ax.twinx()
		print(f"\n\tStarting Random Forest")
		pca_ax.set_title(f"PCA Dimensionality Reduced\n {dataset_name}", fontsize=15, weight='bold')
		pca_ax.set_xlabel("Number of Examples", fontsize=15, weight='heavy')
		pca_ax.set_ylabel("Accuracy", fontsize=15, weight='heavy')
		pca_ax_secondary.set_ylabel("Training Time", fontsize=15, weight='heavy')

		# Baseline
		ln1 = pca_ax.plot(plot_sizes, test_results["Baseline"], marker=".", label="Baseline Accuracy")
		pca_ax.fill_between(plot_sizes, test_results["Baseline"] - base_visualizer.test_scores_std_,
		                    test_results["Baseline"] + base_visualizer.test_scores_std_, alpha=0.2)
		# Training
		ln3 = pca_ax.plot(plot_sizes, train_results["PCA"], marker=".", label="Training Accuracy")
		pca_ax.fill_between(plot_sizes, train_results["PCA"] - pca_visualizer.train_scores_std_,
		                    train_results["PCA"] + pca_visualizer.train_scores_std_, alpha=0.2)
		# Testing
		ln5 = pca_ax.plot(plot_sizes, test_results["PCA"], marker=".", label="Test Accuracy")
		pca_ax.fill_between(plot_sizes, test_results["PCA"] - pca_visualizer.test_scores_std_,
		                    test_results["PCA"] + pca_visualizer.test_scores_std_, alpha=0.2)
		# Run Times
		ln7 = pca_ax_secondary.plot(plot_sizes, train_times["Baseline"], linestyle="--",
		                            label="Baseline Training Time")
		ln8 = pca_ax_secondary.plot(plot_sizes, train_times["PCA"], linestyle="--",
		                            label="Training Time")
		lns = ln1 + ln3 + ln5 + ln7 + ln8
		labs = [l.get_label() for l in lns]
		pca_ax.legend(lns, labs, loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True,
		              shadow=True)
		pca_ax_secondary.grid(alpha=0)
		print(f"\n\tFinished Principal Component Analysis")

		print(f"\n\tStarting Independent Component Analysis")
		ica_ax.set_title(f"ICA Dimensionality Reduced\n {dataset_name}", fontsize=15, weight='bold')
		ica_ax.set_xlabel("Number of Examples", fontsize=15, weight='heavy')
		ica_ax.set_ylabel("Accuracy", fontsize=15, weight='heavy')
		ica_ax_secondary.set_ylabel("Training Time", fontsize=15, weight='heavy')

		# Baseline
		ln1 = ica_ax.plot(plot_sizes, test_results["Baseline"], marker=".", label="Baseline Accuracy")
		ica_ax.fill_between(plot_sizes, test_results["Baseline"] - base_visualizer.test_scores_std_,
		                    test_results["Baseline"] + base_visualizer.test_scores_std_, alpha=0.2)
		# Training
		ln3 = ica_ax.plot(plot_sizes, train_results["ICA"], marker=".", label="Training Accuracy")
		ica_ax.fill_between(plot_sizes, train_results["ICA"] - ica_visualizer.train_scores_std_,
		                    train_results["ICA"] + ica_visualizer.train_scores_std_, alpha=0.2)
		# Testing
		ln5 = ica_ax.plot(plot_sizes, test_results["ICA"], marker=".", label="Test Accuracy")
		ica_ax.fill_between(plot_sizes, test_results["ICA"] - ica_visualizer.test_scores_std_,
		                    test_results["ICA"] + ica_visualizer.test_scores_std_, alpha=0.2)
		# Run Times
		ln7 = ica_ax_secondary.plot(plot_sizes, train_times["Baseline"], linestyle="--",
		                            label="Baseline Training Time")
		ln8 = ica_ax_secondary.plot(plot_sizes, train_times["ICA"], linestyle="--",
		                            label="Training Time")
		lns = ln1 + ln3 + ln5 + ln7 + ln8
		labs = [l.get_label() for l in lns]
		ica_ax.legend(lns, labs, loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True,
		              shadow=True)
		ica_ax_secondary.grid(alpha=0)
		print(f"\n\tFinished Independent Component Analysis")

		print(f"\n\tStarting Randomized Projections")
		rp_ax.set_title(f"Randomized Projection Dimensionality Reduced\n {dataset_name}", fontsize=15, weight='bold')
		rp_ax.set_xlabel("Number of Examples", fontsize=15, weight='heavy')
		rp_ax.set_ylabel("Accuracy", fontsize=15, weight='heavy')
		rp_ax_secondary.set_ylabel("Training Time", fontsize=15, weight='heavy')

		# Baseline
		ln1 = rp_ax.plot(plot_sizes, test_results["Baseline"], marker=".", label="Baseline Accuracy")
		rp_ax.fill_between(plot_sizes, test_results["Baseline"] - base_visualizer.test_scores_std_,
		                   test_results["Baseline"] + base_visualizer.test_scores_std_, alpha=0.2)
		# Training
		ln3 = rp_ax.plot(plot_sizes, train_results["Random_Projection"], marker=".", label="Training Accuracy")
		rp_ax.fill_between(plot_sizes, train_results["Random_Projection"] - rand_proj_visualizer.train_scores_std_,
		                   train_results["Random_Projection"] + rand_proj_visualizer.train_scores_std_, alpha=0.2)
		# Testing
		ln5 = rp_ax.plot(plot_sizes, test_results["Random_Projection"], marker=".", label="Test Accuracy")
		rp_ax.fill_between(plot_sizes, test_results["Random_Projection"] - rand_proj_visualizer.test_scores_std_,
		                   test_results["Random_Projection"] + rand_proj_visualizer.test_scores_std_, alpha=0.2)
		# Run Times
		ln7 = rp_ax_secondary.plot(plot_sizes, train_times["Baseline"], linestyle="--",
		                           label="Baseline Training Time")
		ln8 = rp_ax_secondary.plot(plot_sizes, train_times["Random_Projection"], linestyle="--",
		                           label="Training Time")
		lns = ln1 + ln3 + ln5 + ln7 + ln8
		labs = [l.get_label() for l in lns]
		rp_ax.legend(lns, labs, loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True,
		             shadow=True)
		rp_ax_secondary.grid(alpha=0)
		print(f"\n\tEnding Randomized Projections")

		print(f"\n\tStarting Random Forest")
		rf_ax.set_title(f"Random Forest Dimensionality Reduced\n {dataset_name}", fontsize=15, weight='bold')
		rf_ax.set_xlabel("Number of Examples", fontsize=15, weight='heavy')
		rf_ax.set_ylabel("Accuracy", fontsize=15, weight='heavy')
		rf_ax_secondary.set_ylabel("Training Time", fontsize=15, weight='heavy')

		# Baseline
		ln1 = rf_ax.plot(plot_sizes, test_results["Baseline"], marker=".", label="Baseline Accuracy")
		rf_ax.fill_between(plot_sizes, test_results["Baseline"] - rand_forest_visualizer.test_scores_std_,
		                   test_results["Baseline"] + rand_forest_visualizer.test_scores_std_, alpha=0.2)
		# Training
		ln3 = rf_ax.plot(plot_sizes, train_results["Random_Forest"], marker=".", label="Training Accuracy")
		rf_ax.fill_between(plot_sizes, train_results["Random_Forest"] - rand_forest_visualizer.train_scores_std_,
		                   train_results["Random_Forest"] + rand_forest_visualizer.train_scores_std_, alpha=0.2)
		# Testing
		ln5 = rf_ax.plot(plot_sizes, test_results["Random_Forest"], marker=".", label="Test Accuracy")
		rf_ax.fill_between(plot_sizes, test_results["Random_Forest"] - rand_forest_visualizer.test_scores_std_,
		                   test_results["Random_Forest"] + rand_forest_visualizer.test_scores_std_, alpha=0.2)
		# Run Times
		ln7 = rf_ax_secondary.plot(plot_sizes, train_times["Baseline"], linestyle="--",
		                           label="Baseline Training Time")
		ln8 = rf_ax_secondary.plot(plot_sizes, train_times["Random_Forest"], linestyle="--",
		                           label="Training Time")
		lns = ln1 + ln3 + ln5 + ln7 + ln8
		labs = [l.get_label() for l in lns]
		rf_ax.legend(lns, labs, loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True,
		             shadow=True)
		rf_ax_secondary.grid(alpha=0)
		print(f"\n\tFinished Random Forest")

		with open(f"{os.getcwd()}/Part4/Dimensionality_Reduced_NN_Training_Results.pkl", "wb") as output_file:
			pickle.dump(train_results, output_file)
			output_file.close()

		with open(f"{os.getcwd()}/Part4/Dimensionality_Reduced_NN_Testing_Results.pkl", "wb") as output_file:
			pickle.dump(test_results, output_file)
			output_file.close()

		with open(f"{os.getcwd()}/Part4/Dimensionality_Reduced_NN_Training_Times.pkl", "wb") as output_file:
			pickle.dump(train_times, output_file)
			output_file.close()

		with open(f"{os.getcwd()}/Part4/Dimensionality_Reduced_NN_Predict_Times.pkl", "wb") as output_file:
			pickle.dump(predict_times, output_file)
			output_file.close()

		plt.savefig(f"{os.getcwd()}/Part4/FINAL_NN.png")
		return
	except Exception as run_nn_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'run_nn'", run_nn_exception)


def run_cluster_nn(kmeans_X, kmeans_y, expectation_X, expectation_y, base_line_data_X, base_line_data_y,
           dataset_name="Fashion-MNIST", max_iters=600, alpha=0.001):
	try:
		cv = 10
		n_jobs = 32
		sizes = np.round(np.arange(0.05, 1.01, 0.05), 2)
		plot_sizes = sizes * 8000
		cols = ["Baseline", "KMeans", "Expectation_Maximization"]
		train_results = pd.DataFrame(index=sizes, columns=cols)
		test_results = pd.DataFrame(index=sizes, columns=cols)
		train_times = pd.DataFrame(index=sizes, columns=cols)
		predict_times = pd.DataFrame(index=sizes, columns=cols)

		plt.close("all")
		fig, ax = plt.subplots()
		print(f"\n\tStarting Baseline")
		base_model = MLPClassifier(hidden_layer_sizes=(40,), solver='adam', max_iter=max_iters, alpha=1.0,
		                           verbose=True, random_state=42, warm_start=True)
		base_visualizer = LearningCurve(base_model, cv=cv, scoring='accuracy', train_sizes=sizes, n_jobs=n_jobs,
		                                ax=ax)
		base_visualizer.fit(base_line_data_X, base_line_data_y)
		train_results["Baseline"] = base_visualizer.train_scores_mean_
		test_results["Baseline"] = base_visualizer.test_scores_mean_
		train_times["Baseline"] = base_visualizer.fit_times_mean_
		predict_times["Baseline"] = base_visualizer.predict_times_mean_
		print(f"\n\tFinished Baseline")

		# Kmeans
		print(f"\n\tStarting K-Means Clustering")
		kmeans_nn_clf = MLPClassifier(hidden_layer_sizes=(40,), solver='adam', max_iter=max_iters, alpha=alpha,
		                              verbose=True, random_state=42)
		kmeans_visualizer = LearningCurve(kmeans_nn_clf, cv=cv, scoring='accuracy', train_sizes=sizes,
		                                  n_jobs=n_jobs, ax=ax)
		kmeans_visualizer.fit(kmeans_X, kmeans_y)
		train_results["KMeans"] = kmeans_visualizer.train_scores_mean_
		test_results["KMeans"] = kmeans_visualizer.test_scores_mean_
		train_times["KMeans"] = kmeans_visualizer.fit_times_mean_
		predict_times["KMeans"] = kmeans_visualizer.predict_times_mean_
		print(f"\n\tFinished K-Means Clustering")

		# Expectation Maximization
		print(f"\n\tStarting Expectation Maximization")
		em_nn_clf = MLPClassifier(hidden_layer_sizes=(40,), solver='adam', max_iter=max_iters, alpha=alpha,
		                          verbose=True, random_state=42)
		em_visualizer = LearningCurve(em_nn_clf, cv=cv, scoring='accuracy', train_sizes=sizes,
		                              n_jobs=n_jobs, ax=ax)
		em_visualizer.fit(expectation_X, expectation_y)
		train_results["Expectation_Maximization"] = em_visualizer.train_scores_mean_
		test_results["Expectation_Maximization"] = em_visualizer.test_scores_mean_
		train_times["Expectation_Maximization"] = em_visualizer.fit_times_mean_
		predict_times["Expectation_Maximization"] = em_visualizer.predict_times_mean_
		print(f"\n\tFinished Expectation Maximization")

		plt.close("all")
		plt.style.use("ggplot")
		fig, (kmeans_ax, em_ax) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
		kmeans_ax_secondary = kmeans_ax.twinx()
		em_ax_secondary = em_ax.twinx()

		print(f"\n\tStarting Random Forest")
		kmeans_ax.set_title(f"K-Means Dimensionality Reduced\n {dataset_name}", fontsize=15, weight='bold')
		kmeans_ax.set_xlabel("Number of Examples", fontsize=15, weight='heavy')
		kmeans_ax.set_ylabel("Accuracy", fontsize=15, weight='heavy')
		kmeans_ax_secondary.set_ylabel("Training Time", fontsize=15, weight='heavy')

		# Baseline
		ln1 = kmeans_ax.plot(plot_sizes, test_results["Baseline"], marker=".", label="Baseline Accuracy")
		kmeans_ax.fill_between(plot_sizes, test_results["Baseline"] - base_visualizer.test_scores_std_,
		                       test_results["Baseline"] + base_visualizer.test_scores_std_, alpha=0.2)
		# Training
		ln3 = kmeans_ax.plot(plot_sizes, train_results["KMeans"], marker=".", label="Training Accuracy")
		kmeans_ax.fill_between(plot_sizes, train_results["KMeans"] - kmeans_visualizer.train_scores_std_,
		                       train_results["KMeans"] + kmeans_visualizer.train_scores_std_, alpha=0.2)
		# Testing
		ln5 = kmeans_ax.plot(plot_sizes, test_results["KMeans"], marker=".", label="Test Accuracy")
		kmeans_ax.fill_between(plot_sizes, test_results["KMeans"] - kmeans_visualizer.test_scores_std_,
		                       test_results["KMeans"] + kmeans_visualizer.test_scores_std_, alpha=0.2)
		# Run Times
		ln7 = kmeans_ax_secondary.plot(plot_sizes, train_times["Baseline"], linestyle="--",
		                               label="Baseline Training Time")
		ln8 = kmeans_ax_secondary.plot(plot_sizes, train_times["KMeans"], linestyle="--",
		                               label="Training Time")
		lns = ln1 + ln3 + ln5 + ln7 + ln8
		labs = [l.get_label() for l in lns]
		kmeans_ax.legend(lns, labs, loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True,
		                 shadow=True)
		kmeans_ax_secondary.grid(alpha=0)
		print(f"\n\tFinished KMeans Clustering")

		print(f"\n\tStarting Expectation Maximization")
		em_ax.set_title(f"Expectation Maximization Reduced\n {dataset_name}", fontsize=15, weight='bold')
		em_ax.set_xlabel("Number of Examples", fontsize=15, weight='heavy')
		em_ax.set_ylabel("Accuracy", fontsize=15, weight='heavy')
		em_ax_secondary.set_ylabel("Training Time", fontsize=15, weight='heavy')

		# Baseline
		ln1 = em_ax.plot(plot_sizes, test_results["Baseline"], marker=".", label="Baseline Accuracy")
		em_ax.fill_between(plot_sizes, test_results["Baseline"] - base_visualizer.test_scores_std_,
		                   test_results["Baseline"] + base_visualizer.test_scores_std_, alpha=0.2)
		# Training
		ln3 = em_ax.plot(plot_sizes, train_results["Expectation_Maximization"], marker=".", label="Training Accuracy")
		em_ax.fill_between(plot_sizes, train_results["Expectation_Maximization"] - em_visualizer.train_scores_std_,
		                   train_results["Expectation_Maximization"] + em_visualizer.train_scores_std_, alpha=0.2)
		# Testing
		ln5 = em_ax.plot(plot_sizes, test_results["Expectation_Maximization"], marker=".", label="Test Accuracy")
		em_ax.fill_between(plot_sizes, test_results["Expectation_Maximization"] - em_visualizer.test_scores_std_,
		                   test_results["Expectation_Maximization"] + em_visualizer.test_scores_std_, alpha=0.2)
		# Run Times
		ln7 = em_ax_secondary.plot(plot_sizes, train_times["Baseline"], linestyle="--",
		                           label="Baseline Training Time")
		ln8 = em_ax_secondary.plot(plot_sizes, train_times["Expectation_Maximization"], linestyle="--",
		                           label="Training Time")
		lns = ln1 + ln3 + ln5 + ln7 + ln8
		labs = [l.get_label() for l in lns]
		em_ax.legend(lns, labs, loc="best", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True,
		             shadow=True)
		em_ax_secondary.grid(alpha=0)
		print(f"\n\tFinished Expectation Maximization Analysis")

		with open(f"{os.getcwd()}/Part5/Dimensionality_Reduced_NN_Training_Results.pkl", "wb") as output_file:
			pickle.dump(train_results, output_file)
			output_file.close()

		with open(f"{os.getcwd()}/Part5/Dimensionality_Reduced_NN_Testing_Results.pkl", "wb") as output_file:
			pickle.dump(test_results, output_file)
			output_file.close()

		with open(f"{os.getcwd()}/Part5/Dimensionality_Reduced_NN_Training_Times.pkl", "wb") as output_file:
			pickle.dump(train_times, output_file)
			output_file.close()

		with open(f"{os.getcwd()}/Part5/Dimensionality_Reduced_NN_Predict_Times.pkl", "wb") as output_file:
			pickle.dump(predict_times, output_file)
			output_file.close()

		plt.savefig(f"{os.getcwd()}/Part5/Clustering_FINAL_NN.png")
		return
	except Exception as run_nn_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'run_nn'", run_nn_exception)
