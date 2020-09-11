import copy
import glob
import os
import time

import joblib
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import multiprocessing as mp
import dask.distributed

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import learning_curve, train_test_split


# From Hands on Machine Learning chapter 3 classification
# https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def setup(names=('MNIST',)):
	MYDIR = "dataset"
	CHECK_FOLDER = os.path.isdir(MYDIR)
	
	# If folder doesn't exist, then create it.
	if not CHECK_FOLDER:
		os.makedirs(MYDIR)
		print("created folder : ", MYDIR)
	else:
		print(MYDIR, "folder already exists.")
	
	dataset_directory = "{}/dataset".format(os.getcwd())
	
	files = glob.glob(f"{dataset_directory}/*.feather")
	file_names = set([i.split("\\")[-1]
	                 .split(".")[0] for i in files])
	dataset_path = "{}/dataset".format(os.getcwd())
	dataset = ""
	data = None
	results = {}
	
	for idx in range(len(names)):
		dataset = {"X": "", "y": "", "Full_Dataframe": ""}
		if names[idx].lower() not in file_names:
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


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
	# From Hands on Machine Learning chapter 3 classification
	# https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb
	path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
	print("Saving figure", fig_id)
	if tight_layout:
		plt.tight_layout()
	plt.savefig(path, format=fig_extension, dpi=resolution)
	return


def plot_digit(data):
	# From Hands on Machine Learning chapter 3 classification
	# https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb
	
	image = data.reshape(28, 28)
	plt.imshow(image, cmap=mpl.cm.binary,
	           interpolation="nearest")
	plt.axis("off")
	return


def plot_digits(instances, images_per_row=10, **options):
	# From Hands on Machine Learning chapter 3 classification
	# https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb
	
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


def generate_image_grid(class_names, data_X, data_y, random=False, name="", save_dir=""):
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
		plt.savefig(f"{save_dir}/Image_Grid_{name}.png")
	else:
		plt.savefig(f"{save_dir}/Image_Grid_{name}_random.png")
	return


def combined_generate_image_grid(class_one_names, class_two_names, data1_X, data1_y, data2_X,
                                 data2_y, random=False, save_dir=""):
	cols = 6
	
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
	
	fig = plt.figure(figsize=(20, 10), constrained_layout=True)
	
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
	
	plt.savefig(f"{save_dir}/Image_Grid_Combined_random.png", bbox_inches='tight')
	return


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), save_individual=False, TESTING=False):
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
	if TESTING:
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


def split_data(X, y, Normalize=False):
	train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42)
	train_X.reset_index(drop=True, inplace=True)
	train_y.reset_index(drop=True, inplace=True)
	if Normalize:
		train_X = train_X / 255.0
		train_X = train_X.astype(np.float32)
		test_X = test_X / 255.0
		test_X = test_X.astype(np.float32)
	test_X, valid_X = np.split(test_X.astype(np.float32), 2)
	test_y, valid_y = np.split(test_y.astype(np.uint8), 2)
	test_X.reset_index(drop=True, inplace=True)
	test_y.reset_index(drop=True, inplace=True)
	valid_X.reset_index(drop=True, inplace=True)
	valid_y.reset_index(drop=True, inplace=True)
	return train_X, train_y, valid_X, valid_y, test_X, test_y


def get_cv_result(classifier, train_X, train_y, valid_X, valid_y, test_X, test_y):
	start_time = time.time()
	classifier.fit(train_X, train_y)
	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f"\nTraining Time: {elapsed_time:.6f}s")
	
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
		#endregion
		res = Parallel(n_jobs=cv, backend="threading", verbose=10)(delayed(
			get_cv_result)(classifier=classifiers[i], train_X=train_set_x[i], train_y=train_set_y[i],
		                   valid_X=valid_set_x[i], valid_y=valid_set_y[i], test_X=test_set_x[i],
		                   test_y=test_set_y[i]) for i in range(cv))
		all_data_frames = []
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
	return results, all_cv_results
