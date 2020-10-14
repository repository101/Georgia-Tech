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
import mlrose_hiive
import numpy as np
import pandas as pd
from cycler import cycler
from joblib import delayed, Parallel, parallel_backend
from mlrose_hiive.algorithms.decay import ArithDecay, ExpDecay, GeomDecay
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split, validation_curve
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

import compare_iterations as com_iter

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
			X = StandardScaler().fit_transform(X)
		if normalize and not scale and not minMax and not oneHot:
			X /= max(X.max())
		if minMax and not scale and not normalize:
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


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", valfmt="{x:.2f}",
            textcolors=["white", "black"], threshold=None, x_label=None, y_label=None, title=None,
            filename="", folder=None, cmap="viridis", **kwargs):
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
	
	if not ax:
		ax = plt.gca()
	
	plt.style.use("ggplot")
	# Plot the heatmap
	if title is not None:
		ax.set_title(title, fontsize=15, weight='bold')
	
	im = ax.imshow(data, **kwargs)
	if x_label is not None:
		ax.set_xlabel(x_label, fontsize=15, weight='heavy')
	
	if y_label is not None:
		ax.set_ylabel(y_label, fontsize=15, weight='heavy')
	
	# Create colorbar
	cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
	cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", weight="heavy")
	
	# We want to show all ticks...
	ax.set_xticks(np.arange(data.shape[1]))
	ax.set_yticks(np.arange(data.shape[0]))
	# ... and label them with the respective list entries.
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
	
	final_heatmap = annotate_heatmap(im=im, valfmt=valfmt, textcolors=textcolors, threshold=threshold)
	plt.tight_layout()
	if folder is None:
		plt.savefig(f"{filename}.png")
	else:
		plt.savefig(f"{folder}/{filename}.png")
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


def plot_learning_curve(estimator, title, train_X, train_y, axes=None, ylim=(0.6, 1.01), cv=None,
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
			plt.savefig(f"{os.getcwd()}/_{extra_name}_Learning_Curve.png",
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
			plt.savefig(f"{os.getcwd()}/_{extra_name}_Fit_Times.png",
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
			plt.savefig(f"{os.getcwd()}/_{extra_name}_Fit_Times_Vs_Score.png",
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
		plt.savefig(f"{os.getcwd()}/_{extra_name}__.png", bbox_inches='tight')
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
		plt.savefig(f"{dataset_directory}/Complexity_Analysis/_{model_name}_{orientation}_{extra_name}_.png",
		            bbox_inches='tight')
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
                          folder="SVM", n_classes=10, plot_title=None, dataset_name='MNIST', plot_width=12,
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


def plot_random_optimization(title="", RHC=None, SA=None, GA=None, MIMIC=None):
	if None in {RHC, SA, GA, MIMIC}:
		print("One or more of the passed in results for the various algorithms is empty.")
		return
	plt.close("all")
	
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	return


def plot_single_random_optimization(title="", data=None, xlabel=None, ylabel=None, ):
	if data is None:
		print("Passed in data is empty.")
		return
	plt.close("all")
	
	fig, ax = plt.subplots(2, 2)
	return


def find_best_randomized_hill_climbing(problem, restarts, max_attempts, seed=SEED,
                                       curve=True, verbose=False, maximize=False):
	try:
		df_fitness = pd.DataFrame(index=restarts, columns=max_attempts)
		df_time = pd.DataFrame(index=restarts, columns=max_attempts)
		total_start = time.time()
		best_params = None
		best_state = None
		if maximize:
			best_fitness = -1e10
		else:
			best_fitness = 1e10
		best_fitness_curve = None
		total_iterations = len(restarts) * len(max_attempts)
		count = 0
		print(f"Randomized Hill Climbing Optimization: \n\tTotal Iterations: {total_iterations}")
		
		for restart in restarts:
			for attempt in max_attempts:
				print(f"Max Restarts: {restart}\tMax Attempts: {attempt}")
				start_time = time.time()
				temp_state, \
				temp_fitness, \
				temp_fitness_curve = mlrose_hiive.random_hill_climb(problem=problem, max_attempts=attempt,
				                                                    restarts=restart, curve=curve,
				                                                    random_state=seed)
				end_time = time.time()
				elapsed_time = end_time - start_time
				df_fitness.loc[restart, attempt] = temp_fitness
				df_time.loc[restart, attempt] = elapsed_time
				
				count += 1
				if verbose:
					print("\nRandomized Hill Climb:")
					print(f"\tCurrent Best Fitness: {best_fitness}")
					print(f"\tCurrent Iteration: {count} \n\tRemaining Iterations: {total_iterations - count}"
					      f"\tElapsed Time: {elapsed_time:.4f}s")
					print(f"\tCurrent Restart: {restart} \t Current Attempt: {attempt}")
				if maximize:
					if temp_fitness > best_fitness:
						best_params = {"Restart": restart, "Attempts": attempt}
						best_state = temp_state
						best_fitness = temp_fitness
						best_fitness_curve = temp_fitness_curve
				else:
					if temp_fitness < best_fitness:
						best_params = {"Restart": restart, "Attempts": attempt}
						best_state = temp_state
						best_fitness = temp_fitness
						best_fitness_curve = temp_fitness_curve
		total_end = time.time()
		total_elapsed = total_end - total_start
		print("Finished Randomized Hill Climbing Optimization")
		print(f"Total Time: {total_elapsed}s")
		return best_state, best_fitness, pd.DataFrame(data={"Fitness": best_fitness_curve}), \
		       best_params, df_fitness, df_time
	except Exception as find_best_randomized_hill_climbing_except:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'find_best_randomized_hill_climbing'.\n", find_best_randomized_hill_climbing_except)
		print()


def find_best_genetic_algorithm(problem, pop_sizes, pop_breed_percents, mutation_probs, seed=SEED,
                                curve=True, verbose=False, maximize=False):
	try:
		df_fitness = pd.DataFrame(index=mutation_probs, columns=pop_sizes)
		df_time = pd.DataFrame(index=mutation_probs, columns=pop_sizes)
		total_start = time.time()
		best_params = None
		best_state = None
		if maximize:
			best_fitness = -1e10
		else:
			best_fitness = 1e10
		best_fitness_curve = None
		total_iterations = len(pop_sizes) * len(pop_breed_percents) * len(mutation_probs)
		count = 0
		print(f"Genetic Algorithm Optimization: \n\tTotal Iterations: {total_iterations}")
		
		for pop in pop_sizes:
			for breed_pct in pop_breed_percents:
				for mut_prob in mutation_probs:
					start_time = time.time()
					temp_state, \
					temp_fitness, \
					temp_fitness_curve = mlrose_hiive.genetic_alg(problem=problem, pop_size=pop,
					                                              pop_breed_percent=breed_pct,
					                                              mutation_prob=mut_prob,
					                                              random_state=seed, curve=curve)
					end_time = time.time()
					elapsed_time = end_time - start_time
					df_fitness.loc[mut_prob, pop] = temp_fitness
					df_time.loc[mut_prob, pop] = elapsed_time
					count += 1
					if verbose:
						print("\nGenetic Algorithm:")
						print(f"\tCurrent Best Fitness: {best_fitness}")
						print(f"\tCurrent Iteration: {count} \n\tRemaining Iterations: {total_iterations - count}"
						      f"\tElapsed Time: {elapsed_time:.4f}s")
						print(
							f"\tCurrent Population Size: {pop} \tCurrent Breed Percent: {breed_pct} \t Current Mutation Percent: {mut_prob}")
					if maximize:
						if temp_fitness > best_fitness:
							best_params = {"Population Size": pop, "Breed Percent": breed_pct,
							               "Mutation Probability": mut_prob}
							best_state = temp_state
							best_fitness = temp_fitness
							best_fitness_curve = temp_fitness_curve
					else:
						if temp_fitness < best_fitness:
							best_params = {"Population Size": pop, "Breed Percent": breed_pct,
							               "Mutation Probability": mut_prob}
							best_state = temp_state
							best_fitness = temp_fitness
							best_fitness_curve = temp_fitness_curve
		total_end = time.time()
		total_elapsed = total_end - total_start
		print("Finished Genetic Algorithm Optimization")
		print(f"Total Time: {total_elapsed}s")
		return best_state, best_fitness, pd.DataFrame(data={"Fitness": best_fitness_curve}), best_params, \
		       df_fitness, df_time
	except Exception as find_best_genetic_algorithm_except:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in ' find_best_genetic_algorithm'.\n", find_best_genetic_algorithm_except)
		print()


def find_best_simulated_annealing(problem, decay_schedules, curve=True, seed=SEED,
                                  initial_temps=10. ** np.arange(-2, 2, 1),
                                  min_temps=10. ** np.arange(-3, 1, 1), verbose=False, maximize=False):
	try:
		geom_decays = []
		art_decays = []
		exp_decays = []
		geom_decay_columns, art_decay_columns, exp_decay_columns = None, None, None
		try:
			for init_temp in initial_temps:
				for min_temp in min_temps:
					if init_temp <= min_temp:
						continue
					if len(decay_schedules) >= 3:
						# Exp Decay - decay must be greater than 0
						exp_decay_columns = np.round(((10. ** np.arange(-2, 2, 1)) * 5), 3)
						for exp_decay in exp_decay_columns:
							exp_decays.append(ExpDecay(init_temp=init_temp, min_temp=min_temp, exp_const=exp_decay))
					if len(decay_schedules) >= 2:
						# Arith Decay - decay must be greater than 0 and less than 1 ( does not say must be less than 1 )
						art_decay_columns = np.round(np.unique(np.append(np.arange(0.65, 0.96, 0.05),
						                                                 np.arange(0.96, 0.99, 0.01))), 3)
						for art_decay in art_decay_columns:
							art_decays.append(ArithDecay(init_temp=init_temp, min_temp=min_temp, decay=art_decay))
					if len(decay_schedules) >= 1:
						# Geom Decay - decay must be greater than 0 and less than 1
						geom_decay_columns = np.round(np.arange(0.65, 0.96, 0.05), 3)
						for geo_decay in geom_decay_columns:
							geom_decays.append(GeomDecay(init_temp=init_temp, min_temp=min_temp, decay=geo_decay))
		
		except Exception as run_optimization_tests_except:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)
			print("Exception in 'run_optimization_tests'.\n", run_optimization_tests_except)
			print()
		all_dfs = {}
		decay_columns = []
		decay_types = []
		if geom_decay_columns is not None:
			decay_columns.append(geom_decay_columns)
			decay_types.append(geom_decays)
		if art_decay_columns is not None:
			decay_columns.append(art_decay_columns)
			decay_types.append(art_decays)
		if exp_decay_columns is not None:
			decay_columns.append(exp_decay_columns)
			decay_types.append(exp_decays)
		temp_columns = ["GeomDecay", "ArithDecay", "ExpDecay"]
		
		for decay_iter in range(len(decay_schedules)):
			all_dfs[temp_columns[decay_iter]] = {
				"Correlation DF": pd.DataFrame(index=min_temps,
				                               columns=[f"decay_{i}" for i in decay_columns[decay_iter]],
				                               data=np.zeros(
					                               shape=(min_temps.shape[0], decay_columns[decay_iter].shape[0]))),
				"Runtime DF": pd.DataFrame(index=min_temps, columns=[f"decay_{i}" for i in decay_columns[decay_iter]],
				                           data=np.zeros(
					                           shape=(min_temps.shape[0], decay_columns[decay_iter].shape[0])))
			}
		
		total_start = time.time()
		best_params = None
		best_state = None
		if maximize:
			best_fitness = -1e10
		else:
			best_fitness = 1e10
		best_fitness_curve = None
		total_iterations = len(geom_decays) + len(art_decays) + len(exp_decays)
		count = 0
		print(f"Simulated Annealing Optimization: \n\tTotal Iterations: {total_iterations}")
		
		for dk_type in range(len(decay_types)):
			decay_name = temp_columns[dk_type]
			temp_corr_df = pd.DataFrame()
			temp_time_df = pd.DataFrame()
			for dk in decay_types[dk_type]:
				start_time = time.time()
				temp_state, \
				temp_fitness, \
				temp_fitness_curve = mlrose_hiive.simulated_annealing(problem=problem, schedule=dk, curve=curve,
				                                                      random_state=seed)
				end_time = time.time()
				elapsed_time = end_time - start_time
				if decay_name == temp_columns[2]:
					temp_corr_df.loc[dk.min_temp, f"decay_{dk.exp_const}"] = temp_fitness
					temp_time_df.loc[dk.min_temp, f"decay_{dk.exp_const}"] = elapsed_time
				else:
					temp_corr_df.loc[dk.min_temp, f"decay_{dk.decay}"] = temp_fitness
					temp_time_df.loc[dk.min_temp, f"decay_{dk.decay}"] = elapsed_time
				count += 1
				if verbose:
					print("\nSimulated Annealing:")
					print(f"\tCurrent Best Fitness: {best_fitness}")
					print(f"\tCurrent Iteration: {count} \n\tRemaining Iterations: {total_iterations - count}\n\t"
					      f"\tElapsed Time: {elapsed_time:.4f}s")
					print(f"\tCurrent Decay: {decay_name}")
					if decay_name == temp_columns[2]:
						print(f"\tCurrent min_temp: {dk.min_temp} \t init_temp: {dk.init_temp}\t decay: {dk.exp_const}")
					else:
						print(f"\tCurrent min_temp: {dk.min_temp} \t init_temp: {dk.init_temp}\t decay: {dk.decay}")
				if maximize:
					if temp_fitness > best_fitness:
						best_params = {"Decay Schedule": dk}
						best_state = temp_state
						best_fitness = temp_fitness
						best_fitness_curve = temp_fitness_curve
				else:
					if temp_fitness < best_fitness:
						best_params = {"Decay Schedule": dk}
						best_state = temp_state
						best_fitness = temp_fitness
						best_fitness_curve = temp_fitness_curve
			
			all_dfs[decay_name]["Correlation DF"] = temp_corr_df
			all_dfs[decay_name]["Runtime DF"] = temp_time_df
		total_end = time.time()
		total_elapsed = total_end - total_start
		print("Finished Simulated Annealing Optimization")
		print(f"Total Time: {total_elapsed}s")
		return best_state, best_fitness, pd.DataFrame(data={"Fitness": best_fitness_curve}), best_params, all_dfs
	except Exception as find_best_simulated_annealing_except:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'find_best_simulated_annealing'.\n", find_best_simulated_annealing_except)
		print()


def find_best_mimic(problem, pop_sizes, keep_pcts, curve=True, seed=SEED, verbose=False, maximize=False):
	try:
		df_fitness = pd.DataFrame(index=pop_sizes, columns=keep_pcts)
		df_time = pd.DataFrame(index=pop_sizes, columns=keep_pcts)
		total_start = time.time()
		best_state = None
		best_params = None
		if maximize:
			best_fitness = -1e10
		else:
			best_fitness = 1e10
		best_fitness_curve = None
		total_iterations = len(pop_sizes) * len(keep_pcts)
		count = 0
		print(f"MIMIC Optimization: \n\tTotal Iterations: {total_iterations}")
		for pop in pop_sizes:
			for pct in keep_pcts:
				start_time = time.time()
				temp_state, \
				temp_fitness, \
				temp_fitness_curve = mlrose_hiive.mimic(problem=problem, pop_size=pop, keep_pct=pct,
				                                        curve=curve, random_state=seed, noise=0.05)
				end_time = time.time()
				elapsed_time = end_time - start_time
				df_fitness.loc[pop, pct] = temp_fitness
				df_time.loc[pop, pct] = elapsed_time
				count += 1
				if verbose:
					print("\nMIMIC:")
					print(f"\tCurrent Best Fitness: {best_fitness}")
					print(f"\tCurrent Iteration: {count} \n\tRemaining Iterations: {total_iterations - count}\n\t"
					      f"\tElapsed Time: {elapsed_time:.4f}s")
					print(f"\tPop Size: {pop} \t Keep Percent: {pct}")
				if maximize:
					if temp_fitness > best_fitness:
						best_params = {"Population Size": pop, "Keep Percentage": pct}
						best_fitness = temp_fitness
						best_state = temp_state
						best_fitness_curve = temp_fitness_curve
				else:
					if temp_fitness < best_fitness:
						best_params = {"Population Size": pop, "Keep Percentage": pct}
						best_fitness = temp_fitness
						best_state = temp_state
						best_fitness_curve = temp_fitness_curve
		total_end = time.time()
		total_elapsed = total_end - total_start
		print("Finished MIMIC Optimization")
		print(f"Total Time: {total_elapsed}s")
		return best_state, best_fitness, pd.DataFrame(data={"Fitness": best_fitness_curve}), best_params, \
		       df_fitness, df_time
	except Exception as find_best_mimic_except:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'find_best_mimic'.\n", find_best_mimic_except)
		print()


def compare_iterations(iterations, problem=None, prob_name="TSP", size="s", all_params={}, verbose=False, max_iter=10):
	try:
		if problem is None:
			if isinstance(prob_name, str):
				problem, folder = determine_problem(prob_name=prob_name, size=size)
		
		rhc_results = com_iter.get_rhc_iteration_times(iterations=iterations, problem=copy.deepcopy(problem),
		                                               params=all_params["RHC"],
		                                               verbose=verbose)
		sa_results = com_iter.get_sa_iteration_times(iterations=iterations, problem=copy.deepcopy(problem),
		                                             params=all_params["SA"],
		                                             verbose=verbose)
		ga_results = com_iter.get_ga_iteration_times(iterations=iterations, problem=copy.deepcopy(problem),
		                                             params=all_params["GA"],
		                                             verbose=verbose, max_iter=max_iter)
		mimic_results = com_iter.get_mimic_iteration_times(iterations=iterations, problem=copy.deepcopy(problem),
		                                                   params=all_params["MIMIC"],
		                                                   verbose=verbose)
		results = {"RHC": rhc_results, "GA": ga_results, "SA": sa_results, "MIMIC": mimic_results}
		return results
	except Exception as compare_iterations_except:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'compare_iterations'.\n", compare_iterations_except)
		print()


def find_best_neural_network_gradient_descent(dataset_name="Fashion-MNIST", train_limit=100, learning_rates=None,
                                              iterations=None, verbose=False):
	try:
		folder = "NeuralNetwork/Gradient_Descent"
		check_folder(_directory=folder)
		# Get Data
		gathered_data_fashion = setup(["Fashion-MNIST"])
		fashion_train_X, \
		fashion_train_y, \
		fashion_valid_X, \
		fashion_valid_y, \
		fashion_test_X, \
		fashion_test_y = split_data(gathered_data_fashion[dataset_name]["X"], gathered_data_fashion[dataset_name]["y"],
		                            minMax=True, oneHot=True)
		
		cv_datasets = generate_cv_sets(fashion_train_X, fashion_train_y, cv_sets=5,
		                               train_limit=train_limit, validation_pct=0.2)
		all_avg_validation_acc = []
		all_avg_training_acc = []
		results = {}
		best_valid_accuracy = 0
		best_train_accuracy = 0
		best_network = None
		best_fitness_curve = None
		count = 0
		if learning_rates is None:
			l_rates = 10. ** np.arange(-1, 1, 1)
		else:
			l_rates = learning_rates
		
		if iterations is None:
			n_iterations = np.arange(100, 401, 100).tolist()
		else:
			n_iterations = iterations
		
		total_iterations = len(l_rates) * len(n_iterations)
		temp_learning_rate_vs_iterations_train = pd.DataFrame(columns=[c for c in l_rates],
		                                                      data=np.zeros(
			                                                      shape=(len(n_iterations), len(l_rates))),
		                                                      index=[i for i in n_iterations])
		
		temp_learning_rate_vs_iterations_valid = pd.DataFrame(columns=[c for c in l_rates],
		                                                      data=np.zeros(
			                                                      shape=(len(n_iterations), len(l_rates))),
		                                                      index=[i for i in n_iterations])
		total_start = time.time()
		print(f"Total Iterations: {total_iterations}")
		# Gradient Descent - Baseline should match assignment 1
		for rate in l_rates:
			temp_training_acc = []
			temp_valid_acc = []
			temp_train_time = []
			for iterations in n_iterations:
				start_time = time.time()
				temp_training_acc_container = np.zeros(shape=(len(cv_datasets)))
				temp_validation_acc_container = np.zeros(shape=(len(cv_datasets)))
				temp_time_container = np.zeros(shape=(len(cv_datasets)))
				for cv_idx in range(len(cv_datasets)):
					temp_start_time = time.time()
					temp_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[40], activation='relu', learning_rate=rate,
					                                     max_iters=iterations, algorithm='gradient_descent',
					                                     bias=True, is_classifier=True, early_stopping=True,
					                                     clip_max=5, max_attempts=100, curve=True)
					temp_nn.fit(cv_datasets[f"CV_{cv_idx}"]["Train_X"], cv_datasets[f"CV_{cv_idx}"]["Train_Y"])
					temp_end_time = time.time()
					temp_elapsed_time = temp_end_time - temp_start_time
					temp_time_container[cv_idx] = temp_elapsed_time
					temp_y_pred_train = temp_nn.predict(cv_datasets[f"CV_{cv_idx}"]["Train_X"])
					temp_y_pred_valid = temp_nn.predict(cv_datasets[f"CV_{cv_idx}"]["Validation_X"])
					temp_y_train_acc = accuracy_score(np.argmax(temp_y_pred_train, axis=1),
					                                  np.argmax(cv_datasets[f"CV_{cv_idx}"]["Train_Y"].to_numpy(),
					                                            axis=1))
					temp_y_valid_acc = accuracy_score(np.argmax(temp_y_pred_valid, axis=1),
					                                  np.argmax(
						                                  cv_datasets[f"CV_{cv_idx}"]["Validation_Y"].to_numpy(),
						                                  axis=1))
					temp_training_acc_container[cv_idx] = temp_y_train_acc
					temp_validation_acc_container[cv_idx] = temp_y_valid_acc
					print(f"\t\tCV {cv_idx}: Completed")
				end_time = time.time()
				elapsed_time = end_time - start_time
				temp_avg_train_acc = temp_training_acc_container.mean()
				all_avg_training_acc.append(temp_avg_train_acc)
				temp_training_acc.append(temp_training_acc_container.mean())
				temp_valid_acc.append(temp_validation_acc_container.mean())
				temp_train_time.append(temp_time_container.mean())
				if temp_avg_train_acc > best_train_accuracy:
					best_train_accuracy = temp_avg_train_acc
				temp_avg_valid_acc = temp_validation_acc_container.mean()
				all_avg_validation_acc.append(temp_avg_valid_acc)
				temp_learning_rate_vs_iterations_train.loc[iterations, rate] = temp_avg_train_acc
				temp_learning_rate_vs_iterations_valid.loc[iterations, rate] = temp_avg_valid_acc
				count += 1
				print(f"\tCurrent Iteration: {count} / {total_iterations}")
				if count % 200 == 0 and count > 201:
					print(f"\t\tRemaining Iterations: {total_iterations - count}")
				if temp_avg_valid_acc > best_valid_accuracy:
					best_valid_accuracy = temp_avg_valid_acc
					best_network = temp_nn
					best_fitness_curve = temp_nn.fitness_curve
					if not verbose:
						print(f"\t\tBest Training Accuracy: {best_train_accuracy:.4f}%")
						print(f"\t\tBest Validation Accuracy: {best_valid_accuracy:.4f}%")
						print(f"\t\t\tIteration Time: {elapsed_time:.4f}s")
						print(f"\t\t\tLearning Rate: {rate:.5f}")
						print(f"\t\t\tMax Iterations: {iterations}")
						print(f"\t\t\tIteration Training Accuracy: {temp_avg_train_acc:.4f}%")
						print(f"\t\t\tIteration Validation Accuracy: {temp_avg_valid_acc:.4f}%")
				if verbose:
					print(f"\t\tBest Training Accuracy: {best_train_accuracy:.4f}%")
					print(f"\t\tBest Validation Accuracy: {best_valid_accuracy:.4f}%")
					print(f"\t\t\tIteration Time: {elapsed_time:.4f}s")
					print(f"\t\t\tLearning Rate: {rate:.5f}")
					print(f"\t\t\tMax Iterations: {iterations}")
					print(f"\t\t\tIteration Training Accuracy: {temp_avg_train_acc:.4f}%")
					print(f"\t\t\tIteration Validation Accuracy: {temp_avg_valid_acc:.4f}%")
		
		results["lr_vs_iteration_train"] = temp_learning_rate_vs_iterations_train
		results["lr_vs_iteration_valid"] = temp_learning_rate_vs_iterations_valid
		total_end = time.time()
		total_elapsed = total_end - total_start
		print(f"Total Time for Neural Network Optimization - Gradient Descent: {total_elapsed:.4f}s")
		network_output_file = open(f"{os.getcwd()}/{folder}/Best_Network.pkl", "wb")
		pickle.dump(best_network, network_output_file)
		network_output_file.close()
		results["Best_Network_Object"] = best_network
		temp_df = pd.DataFrame(data={"Fitness": best_fitness_curve})
		results["DataFrame"] = temp_df
		temp_df.to_csv(f"{os.getcwd()}/{folder}/Best_Fitness_Curve.csv", sep=",", index=False)
		with open(f"{folder}/Final_Results_GD.pkl", "wb") as f:
			pickle.dump(results, f)
			f.close()
		return results
	except Exception as find_best_neural_network_gradient_descent_except:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'find_best_neural_network_gradient_descent'.\n",
		      find_best_neural_network_gradient_descent_except)
		print()


def find_best_neural_network_rhc(dataset_name="Fashion-MNIST", train_limit=100, verbose=False, rhc_parameters=None):
	try:
		folder = "NeuralNetwork/Randomized_Hill_Climbing"
		check_folder(_directory=folder)
		# Get Data
		gathered_data_fashion = setup(["Fashion-MNIST"])
		fashion_train_X, \
		fashion_train_y, \
		fashion_valid_X, \
		fashion_valid_y, \
		fashion_test_X, \
		fashion_test_y = split_data(gathered_data_fashion[dataset_name]["X"], gathered_data_fashion[dataset_name]["y"],
		                            minMax=True, oneHot=True)
		
		cv_datasets = generate_cv_sets(fashion_train_X, fashion_train_y, cv_sets=5,
		                               train_limit=train_limit, validation_pct=0.2)
		all_avg_validation_acc = []
		all_avg_training_acc = []
		best_valid_accuracy = 0
		best_train_accuracy = 0
		best_network = None
		best_fitness_curve = None
		count = 0
		results = {}
		time_results = {}
		accuracy_results = {}
		if rhc_parameters is None:
			l_rates = (10. ** np.arange(-1, 2, 1)).tolist()
			n_iterations = [i for i in range(2, 51, 2)]
		else:
			l_rates = rhc_parameters["learning_rates"]
			n_iterations = rhc_parameters["iterations"]
		
		total_iterations = len(l_rates) * len(n_iterations)
		temp_learning_rate_vs_iterations_train = pd.DataFrame(columns=[c for c in l_rates],
		                                                      data=np.zeros(
			                                                      shape=(len(n_iterations), len(l_rates))),
		                                                      index=[i for i in n_iterations])
		temp_learning_rate_vs_iterations_valid = pd.DataFrame(columns=[c for c in l_rates],
		                                                      data=np.zeros(
			                                                      shape=(len(n_iterations), len(l_rates))),
		                                                      index=[i for i in n_iterations])
		
		total_start = time.time()
		print(f"Total Iterations: {total_iterations}")
		# Gradient Descent - Baseline should match assignment 1
		for rate in l_rates:
			temp_training_acc = []
			temp_valid_acc = []
			temp_train_time = []
			for iterations in n_iterations:
				start_time = time.time()
				temp_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[40], activation='relu', learning_rate=rate,
				                                     max_iters=iterations, algorithm='random_hill_climb',
				                                     bias=True, is_classifier=True, early_stopping=True,
				                                     clip_max=5, max_attempts=100,
				                                     curve=True)
				temp_training_acc_container = np.zeros(shape=(len(cv_datasets)))
				temp_validation_acc_container = np.zeros(shape=(len(cv_datasets)))
				temp_time_container = np.zeros(shape=(len(cv_datasets)))
				for cv_idx in range(len(cv_datasets)):
					temp_start_time = time.time()
					temp_nn.fit(cv_datasets[f"CV_{cv_idx}"]["Train_X"], cv_datasets[f"CV_{cv_idx}"]["Train_Y"])
					temp_end_time = time.time()
					temp_elapsed_time = temp_end_time - temp_start_time
					temp_time_container[cv_idx] = temp_elapsed_time
					temp_y_pred_train = temp_nn.predict(cv_datasets[f"CV_{cv_idx}"]["Train_X"])
					temp_y_pred_valid = temp_nn.predict(cv_datasets[f"CV_{cv_idx}"]["Validation_X"])
					
					temp_y_train_acc = accuracy_score(np.argmax(temp_y_pred_train, axis=1),
					                                  np.argmax(
						                                  cv_datasets[f"CV_{cv_idx}"]["Train_Y"].to_numpy(),
						                                  axis=1))
					temp_y_valid_acc = accuracy_score(np.argmax(temp_y_pred_valid, axis=1),
					                                  np.argmax(
						                                  cv_datasets[f"CV_{cv_idx}"][
							                                  "Validation_Y"].to_numpy(),
						                                  axis=1))
					temp_training_acc_container[cv_idx] = temp_y_train_acc
					temp_validation_acc_container[cv_idx] = temp_y_valid_acc
					print(f"\t\tCV {cv_idx}: Completed")
				end_time = time.time()
				elapsed_time = end_time - start_time
				temp_avg_train_acc = temp_training_acc_container.mean()
				all_avg_training_acc.append(temp_avg_train_acc)
				temp_training_acc.append(temp_training_acc_container.mean())
				temp_valid_acc.append(temp_validation_acc_container.mean())
				temp_train_time.append(temp_time_container.mean())
				if temp_avg_train_acc > best_train_accuracy:
					best_train_accuracy = temp_avg_train_acc
				temp_avg_valid_acc = temp_validation_acc_container.mean()
				all_avg_validation_acc.append(temp_avg_valid_acc)
				temp_learning_rate_vs_iterations_train.loc[iterations, rate] = temp_avg_train_acc
				temp_learning_rate_vs_iterations_valid.loc[iterations, rate] = temp_avg_valid_acc
				count += 1
				print(f"\tCurrent Iteration: {count} / {total_iterations}")
				if count % 200 == 0 and count > 201:
					print(f"\t\tRemaining Iterations: {total_iterations - count}")
				if temp_avg_valid_acc > best_valid_accuracy:
					best_valid_accuracy = temp_avg_valid_acc
					best_network = temp_nn
					best_fitness_curve = temp_nn.fitness_curve
					if not verbose:
						print(f"\t\tBest Training Accuracy: {best_train_accuracy:.4f}%")
						print(f"\t\tBest Validation Accuracy: {best_valid_accuracy:.4f}%")
						print(f"\t\t\tIteration Time: {elapsed_time:.4f}s")
						print(f"\t\t\tLearning Rate: {rate:.5f}")
						print(f"\t\t\tMax Iterations: {iterations}")
						print(f"\t\t\tIteration Training Accuracy: {temp_avg_train_acc:.4f}%")
						print(f"\t\t\tIteration Validation Accuracy: {temp_avg_valid_acc:.4f}%")
				if verbose:
					print(f"\t\tBest Training Accuracy: {best_train_accuracy:.4f}%")
					print(f"\t\tBest Validation Accuracy: {best_valid_accuracy:.4f}%")
					print(f"\t\t\tIteration Time: {elapsed_time:.4f}s")
					print(f"\t\t\tLearning Rate: {rate:.5f}")
					print(f"\t\t\tMax Iterations: {iterations}")
					print(f"\t\t\tIteration Training Accuracy: {temp_avg_train_acc:.4f}%")
					print(f"\t\t\tIteration Validation Accuracy: {temp_avg_valid_acc:.4f}%")
			
			accuracy_results[f"lr_{str(rate)}"] = temp_training_acc
			time_results[f"lr_{str(rate)}_time"] = temp_train_time
		results["lr_vs_iteration_train"] = temp_learning_rate_vs_iterations_train
		results["lr_vs_iteration_valid"] = temp_learning_rate_vs_iterations_valid
		total_end = time.time()
		total_elapsed = total_end - total_start
		print(f"Total Time for Neural Network Optimization - Randomized Hill Climb: {total_elapsed:.4f}s")
		network_output_file = open(f"{os.getcwd()}/{folder}/Best_Network.pkl", "wb")
		pickle.dump(best_network, network_output_file)
		network_output_file.close()
		results["Best_Network_Object"] = best_network
		temp_df = pd.DataFrame(data={"Fitness": best_fitness_curve})
		results["DataFrame"] = temp_df
		temp_df.to_csv(f"{os.getcwd()}/{folder}/Best_Fitness_Curve.csv", sep=",", index=False)
		with open(f"{folder}/Final_Results_RHC.pkl", "wb") as f:
			pickle.dump(results, f)
			f.close()
		return results
	except Exception as find_best_neural_network_rhc_except:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'find_best_neural_network_rhc'.\n", find_best_neural_network_rhc_except)


def find_best_neural_network_sa(dataset_name="Fashion-MNIST", train_limit=100, verbose=False,
                                num_iter=20, sa_parameters=None):
	try:
		folder = "NeuralNetwork/Simulated_Annealing"
		check_folder(_directory=folder)
		# Get Data
		gathered_data_fashion = setup(["Fashion-MNIST"])
		fashion_train_X, \
		fashion_train_y, \
		fashion_valid_X, \
		fashion_valid_y, \
		fashion_test_X, \
		fashion_test_y = split_data(gathered_data_fashion[dataset_name]["X"], gathered_data_fashion[dataset_name]["y"],
		                            minMax=True, oneHot=True)
		
		cv_datasets = generate_cv_sets(fashion_train_X, fashion_train_y, cv_sets=5,
		                               train_limit=train_limit, validation_pct=0.2)
		all_avg_validation_acc = []
		all_avg_training_acc = []
		best_valid_accuracy = 0
		best_train_accuracy = 0
		best_network = None
		best_fitness_curve = None
		
		count = 0
		results = {}
		if sa_parameters is None:
			l_rates = np.arange(2000, 19000, 2000)
			n_iterations = np.floor(np.linspace(0, train_limit, num_iter)).tolist()
			temperature_list = 10. ** np.arange(-2, 2, 1)
		else:
			l_rates = sa_parameters["learning_rates"]
			n_iterations = sa_parameters["iterations"]
		
		total_iterations = len(l_rates) * len(n_iterations)
		temp_learning_rate_vs_iterations_train = pd.DataFrame(columns=[c for c in l_rates],
		                                                      data=np.zeros(
			                                                      shape=(len(n_iterations), len(l_rates))),
		                                                      index=[i for i in n_iterations])

		temp_learning_rate_vs_iterations_valid = pd.DataFrame(columns=[c for c in l_rates],
		                                                      data=np.zeros(
			                                                      shape=(len(n_iterations), len(l_rates))),
		                                                      index=[i for i in n_iterations])
		
		total_start = time.time()
		print(f"Total Iterations: {total_iterations}")
		# Gradient Descent - Baseline should match assignment 1
		for rate in l_rates:
			temp_training_acc = []
			temp_valid_acc = []
			temp_train_time = []
			for iterations in n_iterations:
				if iterations <= rate:
					continue
				start_time = time.time()
				temp_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[40], activation='relu', learning_rate=rate,
				                                     max_iters=iterations, algorithm='simulated_annealing',
				                                     schedule=GeomDecay(init_temp=100),
				                                     bias=True, clip_max=10, is_classifier=True,
				                                     early_stopping=True, max_attempts=100, curve=True)
				temp_training_acc_container = np.zeros(shape=(len(cv_datasets)))
				temp_validation_acc_container = np.zeros(shape=(len(cv_datasets)))
				temp_time_container = np.zeros(shape=(len(cv_datasets)))
				for cv_idx in range(len(cv_datasets)):
					temp_start_time = time.time()
					temp_nn.fit(cv_datasets[f"CV_{cv_idx}"]["Train_X"],
					            cv_datasets[f"CV_{cv_idx}"]["Train_Y"])
					temp_end_time = time.time()
					temp_elapsed_time = temp_end_time - temp_start_time
					temp_time_container[cv_idx] = temp_elapsed_time
					temp_y_pred_train = temp_nn.predict(cv_datasets[f"CV_{cv_idx}"]["Train_X"])
					temp_y_pred_valid = temp_nn.predict(cv_datasets[f"CV_{cv_idx}"]["Validation_X"])
					
					temp_y_train_acc = accuracy_score(np.argmax(temp_y_pred_train, axis=1),
					                                  np.argmax(
						                                  cv_datasets[f"CV_{cv_idx}"]["Train_Y"].to_numpy(),
						                                  axis=1))
					temp_y_valid_acc = accuracy_score(np.argmax(temp_y_pred_valid, axis=1),
					                                  np.argmax(
						                                  cv_datasets[f"CV_{cv_idx}"][
							                                  "Validation_Y"].to_numpy(),
						                                  axis=1))
					temp_training_acc_container[cv_idx] = temp_y_train_acc
					temp_validation_acc_container[cv_idx] = temp_y_valid_acc
					
					print(f"\t\tCV {cv_idx}: Completed")
				end_time = time.time()
				elapsed_time = end_time - start_time
				temp_avg_train_acc = temp_training_acc_container.mean()
				all_avg_training_acc.append(temp_avg_train_acc)
				temp_training_acc.append(temp_training_acc_container.mean())
				temp_valid_acc.append(temp_validation_acc_container.mean())
				temp_train_time.append(temp_time_container.mean())
				if temp_avg_train_acc > best_train_accuracy:
					best_train_accuracy = temp_avg_train_acc
				temp_avg_valid_acc = temp_validation_acc_container.mean()
				temp_learning_rate_vs_iterations_train.loc[iterations, rate] = temp_avg_train_acc
				temp_learning_rate_vs_iterations_valid.loc[iterations, rate] = temp_avg_valid_acc
				all_avg_validation_acc.append(temp_avg_valid_acc)
				count += 1
				print(f"\tCurrent Iteration: {count} / {total_iterations}")
				if count % 200 == 0 and count > 201:
					print(f"\t\tRemaining Iterations: {total_iterations - count}")
				if temp_avg_valid_acc > best_valid_accuracy:
					best_valid_accuracy = temp_avg_valid_acc
					best_network = temp_nn
					best_fitness_curve = temp_nn.fitness_curve
					if not verbose:
						print(f"\t\tBest Training Accuracy: {best_train_accuracy:.4f}%")
						print(f"\t\tBest Validation Accuracy: {best_valid_accuracy:.4f}%")
						print(f"\t\t\tIteration Time: {elapsed_time:.4f}s")
						print(f"\t\t\tLearning Rate: {rate:.5f}")
						print(f"\t\t\tMax Iterations: {iterations}")
						print(f"\t\t\tIteration Training Accuracy: {temp_avg_train_acc:.4f}%")
						print(f"\t\t\tIteration Validation Accuracy: {temp_avg_valid_acc:.4f}%")
				if verbose:
					print(f"\t\tBest Training Accuracy: {best_train_accuracy:.4f}%")
					print(f"\t\tBest Validation Accuracy: {best_valid_accuracy:.4f}%")
					print(f"\t\t\tIteration Time: {elapsed_time:.4f}s")
					print(f"\t\t\tLearning Rate: {rate:.5f}")
					print(f"\t\t\tMax Iterations: {iterations}")
					print(f"\t\t\tIteration Training Accuracy: {temp_avg_train_acc:.4f}%")
					print(f"\t\t\tIteration Validation Accuracy: {temp_avg_valid_acc:.4f}%")

		results["lr_vs_iteration_train"] = temp_learning_rate_vs_iterations_train
		results["lr_vs_iteration_valid"] = temp_learning_rate_vs_iterations_valid
		total_end = time.time()
		total_elapsed = total_end - total_start
		print(f"Total Time for Neural Network Optimization - Simulated Annealing: {total_elapsed:.4f}s")
		network_output_file = open(f"{os.getcwd()}/{folder}/Best_Network.pkl", "wb")
		pickle.dump(best_network, network_output_file)
		network_output_file.close()
		results["Best_Network_Object"] = best_network
		temp_df = pd.DataFrame(data={"Fitness": best_fitness_curve})
		results["DataFrame"] = temp_df
		temp_df.to_csv(f"{os.getcwd()}/{folder}/Best_Fitness_Curve.csv", sep=",", index=False)
		with open(f"{folder}/Final_Results_SA.pkl", "wb") as f:
			pickle.dump(results, f)
			f.close()
		return results
	except Exception as find_best_neural_network_sa_except:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'find_best_neural_network_sa'.\n", find_best_neural_network_sa_except)
		print()


def find_best_neural_network_ga(dataset_name="Fashion-MNIST", train_limit=100,
                                verbose=False, num_iter=20, ga_parameters=None, extra_name=None):
	try:
		folder = "NeuralNetwork/Genetic_Algorithm"
		check_folder(_directory=folder)
		# Get Data
		gathered_data_fashion = setup(["Fashion-MNIST"])
		fashion_train_X, \
		fashion_train_y, \
		fashion_valid_X, \
		fashion_valid_y, \
		fashion_test_X, \
		fashion_test_y = split_data(gathered_data_fashion[dataset_name]["X"], gathered_data_fashion[dataset_name]["y"],
		                            minMax=True, oneHot=True)
		
		cv_datasets = generate_cv_sets(fashion_train_X, fashion_train_y, cv_sets=5,
		                               train_limit=train_limit, validation_pct=0.2)
		all_avg_validation_acc = []
		all_avg_training_acc = []
		best_valid_accuracy = 0
		best_train_accuracy = 0
		best_network = None
		best_fitness_curve = None
		count = 0
		results = {}
		if ga_parameters is None:
			l_rates = 10. ** np.arange(-4, 4, 1)
			n_iterations = np.floor(np.linspace(0, train_limit, num_iter)).tolist()
		else:
			l_rates = ga_parameters["learning_rate"]
			n_iterations = ga_parameters["iterations"]
		total_iterations = len(l_rates) * len(n_iterations)
		temp_learning_rate_vs_iterations_train = pd.DataFrame(columns=[c for c in l_rates],
		                                                      data=np.zeros(
			                                                      shape=(len(n_iterations), len(l_rates))),
		                                                      index=[i for i in n_iterations])

		temp_learning_rate_vs_iterations_valid = pd.DataFrame(columns=[c for c in l_rates],
		                                                      data=np.zeros(
			                                                      shape=(len(n_iterations), len(l_rates))),
		                                                      index=[i for i in n_iterations])
		total_start = time.time()
		print(f"Total Iterations: {total_iterations}")
		# Gradient Descent - Baseline should match assignment 1
		for rate in l_rates:
			temp_training_acc = []
			temp_valid_acc = []
			temp_train_time = []
			for iterations in n_iterations:
				if iterations <= rate:
					continue
				start_time = time.time()
				temp_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[40], activation='relu', learning_rate=rate,
				                                     algorithm='genetic_alg', bias=True,
				                                     is_classifier=True, early_stopping=True, clip_max=10,
				                                     max_attempts=5, curve=True)
				temp_training_acc_container = np.zeros(shape=(len(cv_datasets)))
				temp_validation_acc_container = np.zeros(shape=(len(cv_datasets)))
				temp_time_container = np.zeros(shape=(len(cv_datasets)))
				for cv_idx in range(len(cv_datasets)):
					temp_start_time = time.time()
					temp_nn.fit(cv_datasets[f"CV_{cv_idx}"]["Train_X"],
					            cv_datasets[f"CV_{cv_idx}"]["Train_Y"])
					temp_end_time = time.time()
					temp_elapsed_time = temp_end_time - temp_start_time
					temp_time_container[cv_idx] = temp_elapsed_time
					temp_y_pred_train = temp_nn.predict(cv_datasets[f"CV_{cv_idx}"]["Train_X"])
					temp_y_pred_valid = temp_nn.predict(cv_datasets[f"CV_{cv_idx}"]["Validation_X"])
					temp_y_train_acc = accuracy_score(np.argmax(temp_y_pred_train, axis=1),
					                                  np.argmax(
						                                  cv_datasets[f"CV_{cv_idx}"][
							                                  "Train_Y"].to_numpy(),
						                                  axis=1))
					temp_y_valid_acc = accuracy_score(np.argmax(temp_y_pred_valid, axis=1),
					                                  np.argmax(cv_datasets[f"CV_{cv_idx}"][
						                                            "Validation_Y"].to_numpy(),
					                                            axis=1))
					temp_training_acc_container[cv_idx] = temp_y_train_acc
					temp_validation_acc_container[cv_idx] = temp_y_valid_acc
					print(f"\t\tCV {cv_idx}: Completed")
				end_time = time.time()
				elapsed_time = end_time - start_time
				temp_avg_train_acc = temp_training_acc_container.mean()
				all_avg_training_acc.append(temp_avg_train_acc)
				temp_training_acc.append(temp_training_acc_container.mean())
				temp_valid_acc.append(temp_validation_acc_container.mean())
				temp_train_time.append(temp_time_container.mean())
				if temp_avg_train_acc > best_train_accuracy:
					best_train_accuracy = temp_avg_train_acc
				temp_avg_valid_acc = temp_validation_acc_container.mean()
				all_avg_validation_acc.append(temp_avg_valid_acc)
				temp_learning_rate_vs_iterations_train.loc[iterations, rate] = temp_avg_train_acc
				temp_learning_rate_vs_iterations_valid.loc[iterations, rate] = temp_avg_valid_acc
				count += 1
				print(f"\tCurrent Iteration: {count} / {total_iterations}")
				if count % 200 == 0 and count > 201:
					print(f"\t\tRemaining Iterations: {total_iterations - count}")
				if temp_avg_valid_acc > best_valid_accuracy:
					best_valid_accuracy = temp_avg_valid_acc
					best_network = temp_nn
					best_fitness_curve = temp_nn.fitness_curve
					if not verbose:
						print(f"\t\tBest Training Accuracy: {best_train_accuracy:.4f}%")
						print(f"\t\tBest Validation Accuracy: {best_valid_accuracy:.4f}%")
						print(f"\t\t\tIteration Time: {elapsed_time:.4f}s")
						print(f"\t\t\tLearning Rate: {rate:.5f}")
						# print(f"\t\t\tMax Attempts: {attempt}")
						print(f"\t\t\tMax Iterations: {iterations}")
						# print(f"\t\t\tMutation Probability: {prob:.4f}")
						print(f"\t\t\tIteration Training Accuracy: {temp_avg_train_acc:.4f}%")
						print(f"\t\t\tIteration Validation Accuracy: {temp_avg_valid_acc:.4f}%")
				if verbose:
					print(f"\t\tBest Training Accuracy: {best_train_accuracy:.4f}%")
					print(f"\t\tBest Validation Accuracy: {best_valid_accuracy:.4f}%")
					print(f"\t\t\tIteration Time: {elapsed_time:.4f}s")
					print(f"\t\t\tLearning Rate: {rate:.5f}")
					# print(f"\t\t\tMax Attempts: {attempt}")
					print(f"\t\t\tMax Iterations: {iterations}")
					# print(f"\t\t\tMutation Probability: {prob:.4f}")
					print(f"\t\t\tIteration Training Accuracy: {temp_avg_train_acc:.4f}%")
					print(f"\t\t\tIteration Validation Accuracy: {temp_avg_valid_acc:.4f}%")
		
		results["lr_vs_iteration_train"] = temp_learning_rate_vs_iterations_train
		results["lr_vs_iteration_valid"] = temp_learning_rate_vs_iterations_valid
		total_end = time.time()
		total_elapsed = total_end - total_start
		print(f"Total Time for Neural Network Optimization - Genetic Algorithm: {total_elapsed:.4f}s")
		results["Best_Network_Object"] = best_network
		temp_df = pd.DataFrame(data={"Fitness": best_fitness_curve})
		results["DataFrame"] = temp_df
		
		if extra_name is not None:
			temp_df.to_csv(f"{os.getcwd()}/{folder}/Best_Fitness_Curve_{extra_name}.csv", sep=",", index=False)
			with open(f"{os.getcwd()}/{folder}/Best_Network_{extra_name}.pkl", "wb") as f:
				pickle.dump(best_network, f)
				f.close()
			with open(f"{folder}/Final_Results_GA_{extra_name}.pkl", "wb") as f:
				pickle.dump(results, f)
				f.close()
		else:
			temp_df.to_csv(f"{os.getcwd()}/{folder}/Best_Fitness_Curve.csv", sep=",", index=False)
			with open(f"{os.getcwd()}/{folder}/Best_Network_.pkl", "wb") as f:
				pickle.dump(best_network, f)
				f.close()
			with open(f"{folder}/Final_Results_GA.pkl", "wb") as f:
				pickle.dump(results, f)
				f.close()
		return results
	except Exception as find_best_neural_network_ga_except:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'find_best_neural_network_ga'.\n", find_best_neural_network_ga_except)
		print()


def determine_problem(SEED, prob_name="TSP", size="s", maximize=False):
	try:
		folder = ""
		problem = mlrose_hiive.TSPGenerator.generate(seed=SEED, number_of_cities=5, maximize=maximize)
		valid_prob_names = ("TSP", "NQueens", "K Colors", "NN", "tsp", "nqueens", "kcolors",
		                    "nn", "KColors", "knapsack", "KNAPSACK", "flip", "FLIP", "flop", "FLOP",
		                    "flipflop", "FLIPFLOP", "CONTINUOUSPEAKS", "Continuouspeaks", "ContinuousPeaks",
		                    "continuouspeaks", "peaks", "Peaks")
		if prob_name.lower() not in valid_prob_names:
			print("Invalid problem name")
			return
		if size.lower() not in ["s", "m", "l", "xl"]:
			print("Invalid size specified.")
			return
		use_NN = False
		training_data_limit = 0
		
		# region Traveling Salesman
		if prob_name.lower() == "tsp":
			prob_name = "TravelingSalesperson"
			folder = prob_name
			if size.lower() == "s":
				prob_name += "_small"
				problem = mlrose_hiive.generators.TSPGenerator.generate(seed=SEED, number_of_cities=10,
				                                                        maximize=maximize)
			elif size.lower() == "m":
				prob_name += "_medium"
				problem = mlrose_hiive.generators.TSPGenerator.generate(seed=SEED, number_of_cities=15)
			elif size.lower() == "l":
				prob_name += "_large"
				problem = mlrose_hiive.generators.TSPGenerator.generate(seed=SEED, number_of_cities=20)
			elif size.lower() == "xl":
				prob_name += "_extraLarge"
				problem = mlrose_hiive.generators.TSPGenerator.generate(seed=SEED, number_of_cities=30)
			else:
				print("Incorrect size specified")
				return
		# endregion
		
		# region N Queens
		elif prob_name.lower() == "nqueens":
			prob_name = "NQueens"
			folder = prob_name
			if size.lower() == "s":
				prob_name += "_small"
				problem = mlrose_hiive.generators.QueensGenerator.generate(seed=SEED, size=8, maximize=maximize)
			elif size.lower() == "m":
				prob_name += "_medium"
				problem = mlrose_hiive.generators.QueensGenerator.generate(seed=SEED, size=12, maximize=maximize)
			elif size.lower() == "l":
				prob_name += "_large"
				problem = mlrose_hiive.generators.QueensGenerator.generate(seed=SEED, size=16, maximize=maximize)
			elif size.lower() == "xl":
				prob_name += "_extraLarge"
				problem = mlrose_hiive.generators.QueensGenerator.generate(seed=SEED, size=15, maximize=maximize)
			else:
				print("Incorrect size specified")
				return
		# endregion
		
		# region K Colors
		elif prob_name.lower() == "kcolors":
			prob_name = "KColors"
			folder = prob_name
			if size.lower() == "s":
				prob_name += "_small"
				problem = mlrose_hiive.generators.MaxKColorGenerator.generate(seed=SEED, number_of_nodes=15,
				                                                              max_connections_per_node=4,
				                                                              maximize=maximize)
			elif size.lower() == "m":
				prob_name += "_medium"
				problem = mlrose_hiive.generators.MaxKColorGenerator.generate(seed=SEED, number_of_nodes=20,
				                                                              max_connections_per_node=4,
				                                                              maximize=maximize)
			elif size.lower() == "l":
				prob_name += "_large"
				problem = mlrose_hiive.generators.MaxKColorGenerator.generate(seed=SEED, number_of_nodes=25,
				                                                              max_connections_per_node=4,
				                                                              maximize=maximize)
			elif size.lower() == "xl":
				prob_name += "_extraLarge"
				problem = mlrose_hiive.generators.MaxKColorGenerator.generate(seed=SEED, number_of_nodes=25,
				                                                              max_connections_per_node=4,
				                                                              maximize=maximize)
			else:
				print("Incorrect size specified")
				return
		# endregion
		
		# region Four Peaks
		elif prob_name.lower() == "continuouspeaks" or prob_name.lower() == "peaks":
			prob_name = "ContinuousPeaks"
			folder = prob_name
			if size.lower() == "s":
				prob_name += "_small"
				problem = mlrose_hiive.generators.ContinuousPeaksGenerator.generate(seed=SEED, size=20,
				                                                                    maximize=maximize)
			elif size.lower() == "m":
				prob_name += "_medium"
				problem = mlrose_hiive.generators.ContinuousPeaksGenerator.generate(seed=SEED, size=40,
				                                                                    maximize=maximize)
			elif size.lower() == "l":
				prob_name += "_large"
				problem = mlrose_hiive.generators.ContinuousPeaksGenerator.generate(seed=SEED, size=80,
				                                                                    maximize=maximize)
			elif size.lower() == "xl":
				prob_name += "_extraLarge"
				problem = mlrose_hiive.generators.ContinuousPeaksGenerator.generate(seed=SEED, size=25,
				                                                                    maximize=maximize)
			else:
				print("Incorrect size specified")
				return
		# endregion
		
		# region Knapsacks
		elif prob_name.lower() == "knapsack":
			# Maximize not specified b/c by default fit func uses maximization instead of minimization
			prob_name = "Knapsack"
			folder = prob_name
			if size.lower() == "s":
				prob_name += "_small"
				problem = mlrose_hiive.generators.KnapsackGenerator.generate(seed=SEED, number_of_items_types=30,
				                                                             maximize=maximize)
			elif size.lower() == "m":
				prob_name += "_medium"
				problem = mlrose_hiive.generators.KnapsackGenerator.generate(seed=SEED, number_of_items_types=40,
				                                                             maximize=maximize)
			elif size.lower() == "l":
				prob_name += "_large"
				problem = mlrose_hiive.generators.KnapsackGenerator.generate(seed=SEED, number_of_items_types=50,
				                                                             maximize=maximize)
			elif size.lower() == "xl":
				prob_name += "_extraLarge"
				problem = mlrose_hiive.generators.KnapsackGenerator.generate(seed=SEED, number_of_items_types=25,
				                                                             maximize=maximize)
			else:
				print("Incorrect size specified")
				return
		# endregion
		
		# region FlipFlop
		elif prob_name.lower() == "flip" or prob_name.lower() == "flop" or prob_name.lower() == "flipflop":
			prob_name = "FlipFlop"
			folder = prob_name
			if size.lower() == "s":
				prob_name += "_small"
				problem = mlrose_hiive.generators.FlipFlopGenerator.generate(seed=SEED, size=30, maximize=maximize)
			elif size.lower() == "m":
				prob_name += "_medium"
				problem = mlrose_hiive.generators.FlipFlopGenerator.generate(seed=SEED, size=40, maximize=maximize)
			elif size.lower() == "l":
				prob_name += "_large"
				problem = mlrose_hiive.generators.FlipFlopGenerator.generate(seed=SEED, size=90, maximize=maximize)
			elif size.lower() == "xl":
				prob_name += "_extraLarge"
				problem = mlrose_hiive.generators.FlipFlopGenerator.generate(seed=SEED, size=50, maximize=maximize)
			else:
				print("Incorrect size specified")
				return
		# endregion
		
		# region Neural Network
		elif prob_name.lower() == "nn":
			use_NN = True
			prob_name = "NeuralNetwork"
			folder = prob_name
			if size.lower() == "s":
				prob_name += "_small"
				training_data_limit = 500
			elif size.lower() == "m":
				prob_name += "_medium"
				training_data_limit = 1000
			elif size.lower() == "l":
				prob_name += "_large"
				training_data_limit = 2000
			elif size.lower() == "xl":
				prob_name += "_extraLarge"
				training_data_limit = 4000
			else:
				print("Incorrect size specified")
				return
		# endregion
		return problem, folder
	except Exception as determine_problem_except:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'determine_problem'.\n", determine_problem_except)
		print()


def find_best_parameters(prob_name, iterations, maximize, max_attempts, parameters, size, cv=1, gen_curves=True):
	try:
		rhc_restarts = np.zeros(shape=(cv,))
		rhc_attempts = np.zeros(shape=(cv,))
		rhc_iters = np.zeros(shape=(cv,))
		rhc_fit = 0
		
		sa_decay = np.zeros(shape=(cv, cv), dtype=object)
		sa_attempts = np.zeros(shape=(cv,))
		sa_iters = np.zeros(shape=(cv,))
		sa_fit = 0
		
		ga_pop_size = np.zeros(shape=(cv,))
		ga_mut_rate = np.zeros(shape=(cv,))
		ga_iters = np.zeros(shape=(cv,))
		ga_attempts = np.zeros(shape=(cv,))
		ga_fit = 0
		
		mimic_keep_percent = np.zeros(shape=(cv,))
		mimic_pop_size = np.zeros(shape=(cv,))
		mimic_iters = np.zeros(shape=(cv,))
		mimic_attempts = np.zeros(shape=(cv,))
		mimic_fit = 0
		
		for i in range(cv):
			problem, folder = determine_problem(prob_name=prob_name, size=size, maximize=maximize,
			                                    SEED=int(np.round(time.time())))
			check_folder(_directory=folder)
			temp_rhc = mlrose_hiive.runners.RHCRunner(problem=problem, seed=int(np.round(time.time())),
			                                          generate_curves=False,
			                                          experiment_name=f"RandomHillClimb_{prob_name}_{size}",
			                                          iteration_list=parameters["RHC"]["iterations"],
			                                          max_attempts=max_attempts,
			                                          restart_list=parameters["RHC"]["restart_list"],
			                                          maximize=maximize, return_results=True)
			rhc_run_stats, rhc_run_curves = temp_rhc.run()
			rhc_fit = np.max(rhc_run_stats["Fitness"])
			rhc_restarts[i] = rhc_run_stats.iloc[np.argmax(rhc_run_stats["Fitness"])]["Restarts"]
			rhc_iters[i] = rhc_run_stats.iloc[np.argmax(rhc_run_stats["Fitness"])]["max_iters"]
			rhc_attempts[i] = max_attempts
			np.savetxt(f"{folder}/RandomHillClimb_Best_Restarts_{size}_{size}_{size}.csv", X=rhc_restarts,
			           delimiter=",", fmt="%.2f")
			np.savetxt(f"{folder}/RandomHillClimb_Best_Iteration_{size}_{size}_{size}.csv", X=rhc_iters, delimiter=",",
			           fmt="%.2f")
			np.savetxt(f"{folder}/RandomHillClimb_Best_Attempts_{size}_{size}_{size}.csv", X=rhc_attempts,
			           delimiter=",", fmt="%.2f")
			
			problem, folder = determine_problem(prob_name=prob_name, size=size, maximize=maximize,
			                                    SEED=int(np.round(time.time())))
			temp_sa = mlrose_hiive.runners.SARunner(problem=problem, seed=int(np.round(time.time())),
			                                        generate_curves=gen_curves,
			                                        experiment_name=f"SimulatedAnnealing_{prob_name}_{size}",
			                                        max_attempts=parameters["SA"]["max_attempts"],
			                                        temperature_list=parameters["SA"]["temperature_list"],
			                                        iteration_list=parameters["SA"]["iterations"],
			                                        maximize=maximize,
			                                        decay_list=[mlrose_hiive.GeomDecay, mlrose_hiive.ArithDecay],
			                                        decay_pcts=[0.95, 0.97, 0.99], return_results=True)
			sa_run_stats, sa_run_curves = temp_sa.run()
			sa_fit = np.max(sa_run_stats["Fitness"])
			sa_decay[i, 0] = sa_run_stats.iloc[np.argmax(sa_run_stats["Fitness"])]["Temperature"]
			sa_decay[i, 1] = sa_run_stats.iloc[np.argmax(sa_run_stats["Fitness"])]["Fitness"]
			sa_iters[i] = sa_run_stats.iloc[np.argmax(sa_run_stats["Fitness"])]["max_iters"]
			sa_attempts[i] = max_attempts
			with open(f"{folder}/SimulatedAnnealing_Best_Decay_{size}_{size}_{size}.pkl", "wb") as f:
				np.save(f, sa_decay)
				f.close()
			np.savetxt(f"{folder}/SimulatedAnnealing_Best_Iteration_{size}_{size}_{size}.csv", X=sa_iters,
			           delimiter=",", fmt="%.2f")
			np.savetxt(f"{folder}/SimulatedAnnealing_Best_Attempts_{size}_{size}_{size}.csv", X=sa_iters, delimiter=",",
			           fmt="%.2f")
			
			problem, folder = determine_problem(prob_name=prob_name, size=size, maximize=maximize,
			                                    SEED=int(np.round(time.time())))
			
			temp_ga = mlrose_hiive.runners.GARunner(problem=problem, seed=int(np.round(time.time())),
			                                        generate_curves=gen_curves,
			                                        experiment_name=f"GeneticAlgorithm_{prob_name}_{size}",
			                                        iteration_list=parameters["GA"]["iterations"],
			                                        max_attempts=parameters["GA"]["max_attempts"],
			                                        population_sizes=parameters["GA"]["population_sizes"],
			                                        mutation_rates=parameters["GA"]["mutation_rates"],
			                                        maximize=maximize, return_results=True)
			ga_run_stats, ga_run_curves = temp_ga.run()
			ga_fit = np.max(ga_run_stats["Fitness"])
			ga_temp_best = ga_run_stats.iloc[np.argmax(ga_run_stats["Fitness"])]
			ga_pop_size[i] = ga_temp_best["Population Size"]
			ga_mut_rate[i] = ga_temp_best["Mutation Rate"]
			ga_iters[i] = ga_temp_best["max_iters"]
			ga_attempts[i] = max_attempts
			np.savetxt(f"{folder}/GeneticAlgorithm_Best_Pop_Size_{size}_{size}_{size}.csv", X=ga_pop_size,
			           delimiter=",", fmt="%.2f")
			np.savetxt(f"{folder}/GeneticAlgorithm_Best_Mut_Rate_{size}_{size}_{size}.csv", X=ga_mut_rate,
			           delimiter=",", fmt="%.2f")
			np.savetxt(f"{folder}/GeneticAlgorithm_Best_Iteration_{size}_{size}_{size}.csv", X=ga_iters, delimiter=",",
			           fmt="%.2f")
			np.savetxt(f"{folder}/GeneticAlgorithm_Best_Attempts_{size}_{size}_{size}.csv", X=ga_attempts,
			           delimiter=",", fmt="%.2f")
			
			problem, folder = determine_problem(prob_name=prob_name, size=size, maximize=maximize,
			                                    SEED=int(np.round(time.time())))
			mimic = mlrose_hiive.runners.MIMICRunner(problem=problem, seed=int(np.round(time.time())),
			                                         generate_curves=gen_curves,
			                                         experiment_name=f"MIMIC_{prob_name}_{size}",
			                                         iteration_list=parameters["MIMIC"]["iterations"],
			                                         max_attempts=parameters["MIMIC"]["max_attempts"],
			                                         population_sizes=parameters["MIMIC"]["population_sizes"],
			                                         keep_percent_list=parameters["MIMIC"]["keep_percent_list"],
			                                         maximize=maximize, use_fast_mimic=True, noise=0.05,
			                                         return_results=True)
			mimic_run_stats, mimic_run_curves = mimic.run()
			mimic_fit = np.max(mimic_run_stats["Fitness"])
			mimic_temp_best = mimic_run_stats.iloc[np.argmax(mimic_run_stats["Fitness"])]
			mimic_iters[i] = mimic_temp_best["max_iters"]
			mimic_keep_percent[i] = mimic_temp_best["Keep Percent"]
			mimic_pop_size[i] = mimic_temp_best["Population Size"]
			mimic_attempts[i] = max_attempts
			np.savetxt(f"{folder}/MIMIC_Best_Pop_Size_{size}_{size}_{size}.csv", X=mimic_pop_size, delimiter=",",
			           fmt="%.2f")
			np.savetxt(f"{folder}/MIMIC_Best_Keep_Percent_{size}_{size}_{size}.csv", X=mimic_keep_percent,
			           delimiter=",", fmt="%.2f")
			np.savetxt(f"{folder}/MIMIC_Best_Iteration_{size}_{size}_{size}.csv", X=mimic_iters, delimiter=",",
			           fmt="%.2f")
			np.savetxt(f"{folder}/MIMIC_Best_Attempts_{size}_{size}_{size}.csv", X=mimic_attempts, delimiter=",",
			           fmt="%.2f")
		
		if cv == 1:
			rhc_params = {"restarts": rhc_restarts[0], "max_iters": rhc_iters[0], "max_attempts": rhc_attempts[0]}
			sa_params = {"schedule": sa_decay[0], "max_iters": sa_iters[0], "max_attempts": sa_attempts[0]}
			ga_params = {"pop_size": ga_pop_size[0], "mutation_prob": ga_mut_rate[0], "max_iters": ga_iters[0],
			             "max_attempts": ga_attempts[0]}
			mimic_params = {"pop_size": mimic_pop_size[0], "keep_pct": mimic_keep_percent[0],
			                "max_attempts": mimic_attempts[0], "max_iters": mimic_iters[0]}
		else:
			rhc_params = {"restarts": int(np.round(rhc_restarts.mean())),
			              "max_iters": int(np.round(rhc_iters.mean())),
			              "max_attempts": int(np.round(rhc_restarts.mean()))}
			sa_params = {"schedule": sa_decay[np.argmax(sa_decay[:, 1]), 0],
			             "max_iters": int(np.round(sa_iters.mean())),
			             "max_attempts": int(np.round((sa_attempts.mean())))}
			ga_params = {"pop_size": int(np.round((ga_pop_size.mean()))),
			             "mutation_prob": np.round(ga_mut_rate.mean(), 2),
			             "max_iters": int(np.round((ga_iters.mean()))),
			             "max_attempts": int(np.round((ga_attempts.mean())))}
			mimic_params = {"pop_size": int(np.round(mimic_pop_size.mean(), 2)),
			                "keep_pct": np.round(mimic_keep_percent.mean(), 2),
			                "max_attempts": int(np.round((mimic_attempts.mean()))),
			                "max_iters": int(np.round((mimic_iters.mean())))}
		best_params = {"RHC": rhc_params, "SA": sa_params, "GA": ga_params, "MIMIC": mimic_params}
		best_fit = {"RHC": rhc_fit, "SA": sa_fit, "GA": ga_fit, "MIMIC": mimic_fit}
		return best_params, best_fit
	except Exception as find_best_parameters_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in find_best_parameters.", find_best_parameters_exception)


def run_evaluations(params, prob_name, eval_sizes=["s", "m", "l"], eval_cv=5, max_attempts=100, max_iters=5000,
                    change_pop=False, reset_pop=True):
	try:
		result_frame = {"Fitness": 0, "Curve": 0, "RunTime": 0, "Iterations": 0, "Alt_runtime": 0}
		folder = prob_name
		
		# region Randomized Hill Climbing
		rhc_results = {"s": copy.deepcopy(result_frame),
		               "m": copy.deepcopy(result_frame),
		               "l": copy.deepcopy(result_frame)}
		
		for _size in eval_sizes:
			print(f"Starting RHC: {_size}")
			temp_problem, _folder = determine_problem(prob_name=prob_name, size=_size, maximize=True,
			                                          SEED=int(np.round(time.time())))
			best_parameters = read_parameters_from_file(folder=_folder, cv=2, size=_size, change_pop=change_pop,
			                                            reset_pop=reset_pop)
			rhc_parameters = best_parameters["RHC"]
			print(f"Current parameters: ", rhc_parameters)
			temp_times = np.zeros(shape=(eval_cv,))
			temp_fitness_container = np.zeros(shape=(eval_cv,))
			temp_fitness_curve_container = np.zeros(shape=(rhc_parameters["max_iters"] + 1, eval_cv))
			temp_best_fitness = 0
			temp_iteration_tracker = np.zeros(shape=(eval_cv,), dtype=object)
			temp_time_tracker = np.zeros(shape=(rhc_parameters["max_iters"] + 1, eval_cv))
			for _iter in range(eval_cv):
				temp_problem, folder = determine_problem(prob_name=prob_name, size=_size, maximize=True,
				                                         SEED=int(np.round(time.time())))
				temp_start_time = time.time()
				
				temp_state, \
				temp_fitness, \
				temp_fitness_curve, \
				temp_timing_iteration = mlrose_hiive.random_hill_climb(problem=temp_problem,
				                                                       max_attempts=rhc_parameters["max_attempts"],
				                                                       restarts=rhc_parameters["restarts"], curve=True,
				                                                       random_state=int(np.round(time.time())),
				                                                       max_iters=rhc_parameters["max_iters"],
				                                                       return_results=True)
				temp_end_time = time.time()
				temp_elapsed_time = temp_end_time - temp_start_time
				temp_times[_iter] = temp_elapsed_time
				temp_fitness_container[_iter] = temp_fitness
				temp_fitness_curve_container[0:temp_fitness_curve.shape[0], _iter] = temp_fitness_curve
				temp_iteration_tracker[_iter] = temp_timing_iteration["call_tracker"]
				temp_time_tracker[0:temp_timing_iteration["run_times"].shape[0], _iter] = \
					np.sum(temp_timing_iteration["run_times"], axis=1)
				if temp_fitness > temp_best_fitness:
					temp_best_fitness = temp_fitness
				print(f"\tElapsed Time: {temp_elapsed_time:.4f}s\tCurrent Best Fitness: {temp_best_fitness}")
			
			limit_idx = np.max(np.argwhere(temp_fitness_curve_container > 0)[:, 0])
			rhc_results[_size]["RunTime"] = temp_times
			rhc_results[_size]["Curve"] = temp_fitness_curve_container[:limit_idx + 1, :]
			rhc_results[_size]["Fitness"] = temp_fitness_container
			
			rhc_results[_size]["Avg_Iterations"] = np.mean(temp_iteration_tracker)
			rhc_results[_size]["All_Iterations"] = temp_iteration_tracker
			rhc_results[_size]["Alt_runtime"] = temp_time_tracker[:limit_idx + 1, :]
			rhc_results[_size]["Avg_runtime"] = np.mean(rhc_results[_size]["Alt_runtime"], axis=1)
		
		with open(f"{os.getcwd()}/{folder}/All_RHC_Results.pkl", "wb") as temp_file:
			pickle.dump(rhc_results, temp_file)
			temp_file.close()
		
		# endregion
		
		# region Simulated Annealing
		sa_results = {"s": copy.deepcopy(result_frame),
		              "m": copy.deepcopy(result_frame),
		              "l": copy.deepcopy(result_frame)}
		for _size in eval_sizes:
			print(f"Starting Simulated Annealing: {_size}")
			temp_problem, _folder = determine_problem(prob_name=prob_name, size=_size, maximize=True,
			                                          SEED=int(np.round(time.time())))
			best_parameters = read_parameters_from_file(folder=_folder, cv=2, size=_size, change_pop=change_pop,
			                                            reset_pop=reset_pop)
			sa_parameters = best_parameters["SA"]
			print(f"Current parameters: ", sa_parameters)
			temp_times = np.zeros(shape=(eval_cv,))
			temp_fitness_container = np.zeros(shape=(eval_cv,))
			temp_fitness_curve_container = np.zeros(shape=(sa_parameters["max_iters"] + 1, eval_cv))
			temp_best_fitness = 0
			temp_iteration_tracker = np.zeros(shape=(eval_cv,), dtype=object)
			temp_time_tracker = np.zeros(shape=(sa_parameters["max_iters"] + 1, eval_cv))
			for _iter in range(eval_cv):
				temp_problem, folder = determine_problem(prob_name=prob_name, size=_size, maximize=True,
				                                         SEED=int(np.round(time.time())))
				temp_start_time = time.time()
				
				temp_state, \
				temp_fitness, \
				temp_fitness_curve, \
				temp_timing_iteration = mlrose_hiive.simulated_annealing(problem=temp_problem,
				                                                         max_attempts=sa_parameters["max_attempts"],
				                                                         schedule=sa_parameters["schedule"], curve=True,
				                                                         random_state=int(np.round(time.time())),
				                                                         max_iters=sa_parameters["max_iters"],
				                                                         return_results=True)
				temp_end_time = time.time()
				temp_elapsed_time = temp_end_time - temp_start_time
				temp_times[_iter] = temp_elapsed_time
				temp_fitness_container[_iter] = temp_fitness
				temp_fitness_curve_container[0:temp_fitness_curve.shape[0], _iter] = temp_fitness_curve
				temp_iteration_tracker[_iter] = temp_timing_iteration["call_tracker"]
				temp_time_tracker[0:temp_timing_iteration["run_times"].shape[0], _iter] = \
					temp_timing_iteration["run_times"]
				if temp_fitness > temp_best_fitness:
					temp_best_fitness = temp_fitness
				print(f"\tElapsed Time: {temp_elapsed_time:.4f}s\tCurrent Best Fitness: {temp_best_fitness}")
			
			limit_idx = np.max(np.argwhere(temp_fitness_curve_container > 0)[:, 0])
			sa_results[_size]["RunTime"] = temp_times
			sa_results[_size]["Curve"] = temp_fitness_curve_container[:limit_idx + 1, :]
			sa_results[_size]["Fitness"] = temp_fitness_container
			
			sa_results[_size]["Avg_Iterations"] = np.mean(temp_iteration_tracker)
			sa_results[_size]["All_Iterations"] = temp_iteration_tracker
			sa_results[_size]["Alt_runtime"] = temp_time_tracker[:limit_idx + 1, :]
			sa_results[_size]["Avg_runtime"] = np.mean(sa_results[_size]["Alt_runtime"], axis=1)
		
		with open(f"{os.getcwd()}/{folder}/All_SA_Results.pkl", "wb") as temp_file:
			pickle.dump(sa_results, temp_file)
			temp_file.close()
		# endregion
		
		# region Genetic Algorithm
		ga_results = {"s": copy.deepcopy(result_frame),
		              "m": copy.deepcopy(result_frame),
		              "l": copy.deepcopy(result_frame)}
		for _size in eval_sizes:
			print(f"Starting Genetic Algorithm: {_size}")
			temp_problem, _folder = determine_problem(prob_name=prob_name, size=_size, maximize=True,
			                                          SEED=int(np.round(time.time())))
			best_parameters = read_parameters_from_file(folder=_folder, cv=2, size=_size, change_pop=change_pop,
			                                            reset_pop=reset_pop)
			ga_parameters = best_parameters["GA"]
			print(f"Current parameters: ", ga_parameters)
			temp_times = np.zeros(shape=(eval_cv,))
			temp_fitness_container = np.zeros(shape=(eval_cv,))
			temp_fitness_curve_container = np.zeros(shape=(ga_parameters["max_iters"] + 1, eval_cv))
			temp_best_fitness = 0
			temp_iteration_tracker = np.zeros(shape=(eval_cv,), dtype=object)
			temp_time_tracker = np.zeros(shape=(ga_parameters["max_iters"] + 1, eval_cv))
			for _iter in range(eval_cv):
				temp_problem, folder = determine_problem(prob_name=prob_name, size=_size, maximize=True,
				                                         SEED=int(np.round(time.time())))
				temp_start_time = time.time()
				
				temp_state, \
				temp_fitness, \
				temp_fitness_curve, \
				temp_timing_iteration = mlrose_hiive.genetic_alg(problem=temp_problem,
				                                                 max_attempts=ga_parameters["max_attempts"], curve=True,
				                                                 pop_size=ga_parameters["pop_size"],
				                                                 mutation_prob=ga_parameters["mutation_prob"],
				                                                 random_state=int(np.round(time.time())),
				                                                 max_iters=ga_parameters["max_iters"],
				                                                 return_results=True)
				temp_end_time = time.time()
				temp_elapsed_time = temp_end_time - temp_start_time
				temp_times[_iter] = temp_elapsed_time
				temp_fitness_container[_iter] = temp_fitness
				temp_fitness_curve_container[0:temp_fitness_curve.shape[0], _iter] = temp_fitness_curve
				temp_iteration_tracker[_iter] = temp_timing_iteration["call_tracker"]
				temp_time_tracker[0:temp_timing_iteration["run_times"].shape[0], _iter] = \
					temp_timing_iteration["run_times"][:, 0]
				if temp_fitness > temp_best_fitness:
					temp_best_fitness = temp_fitness
				print(f"\tElapsed Time: {temp_elapsed_time:.4f}s\tCurrent Best Fitness: {temp_best_fitness}")
			
			limit_idx = np.max(np.argwhere(temp_fitness_curve_container > 0)[:, 0])
			ga_results[_size]["RunTime"] = temp_times
			ga_results[_size]["Curve"] = temp_fitness_curve_container[:limit_idx + 1, :]
			ga_results[_size]["Fitness"] = temp_fitness_container
			
			ga_results[_size]["Avg_Iterations"] = np.mean(temp_iteration_tracker)
			ga_results[_size]["All_Iterations"] = temp_iteration_tracker
			ga_results[_size]["Alt_runtime"] = temp_time_tracker[:limit_idx, :]
			ga_results[_size]["Avg_runtime"] = np.mean(ga_results[_size]["Alt_runtime"], axis=1)
		
		with open(f"{os.getcwd()}/{folder}/All_GA_Results.pkl", "wb") as temp_file:
			pickle.dump(ga_results, temp_file)
			temp_file.close()
		# endregion
		
		# region MIMIC
		mimic_results = {"s": copy.deepcopy(result_frame),
		                 "m": copy.deepcopy(result_frame),
		                 "l": copy.deepcopy(result_frame)}
		for _size in eval_sizes:
			print(f"Starting MIMIC: {_size}")
			temp_problem, _folder = determine_problem(prob_name=prob_name, size=_size, maximize=True,
			                                          SEED=int(np.round(time.time())))
			best_parameters = read_parameters_from_file(folder=_folder, cv=2, size=_size, change_pop=change_pop,
			                                            reset_pop=reset_pop)
			mimic_parameters = best_parameters["MIMIC"]
			print(f"Current parameters: ", mimic_parameters)
			temp_times = np.zeros(shape=(eval_cv,))
			temp_fitness_container = np.zeros(shape=(eval_cv,))
			temp_fitness_curve_container = np.zeros(shape=(mimic_parameters["max_iters"] + 1, eval_cv))
			temp_best_fitness = 0
			temp_iteration_tracker = np.zeros(shape=(eval_cv,), dtype=object)
			temp_time_tracker = np.zeros(shape=(mimic_parameters["max_iters"] + 1, eval_cv))
			for _iter in range(eval_cv):
				temp_problem, folder = determine_problem(prob_name=prob_name, size=_size, maximize=True,
				                                         SEED=int(np.round(time.time())))
				temp_start_time = time.time()
				
				temp_state, \
				temp_fitness, \
				temp_fitness_curve, \
				temp_timing_iteration = mlrose_hiive.mimic(problem=temp_problem,
				                                           max_attempts=mimic_parameters["max_attempts"], curve=True,
				                                           random_state=int(np.round(time.time())),
				                                           max_iters=mimic_parameters["max_iters"],
				                                           pop_size=mimic_parameters["pop_size"],
				                                           keep_pct=mimic_parameters["keep_pct"], noise=0.05,
				                                           return_results=True)
				temp_end_time = time.time()
				temp_elapsed_time = temp_end_time - temp_start_time
				temp_times[_iter] = temp_elapsed_time
				temp_fitness_container[_iter] = temp_fitness
				temp_fitness_curve_container[0:temp_fitness_curve.shape[0], _iter] = temp_fitness_curve
				temp_iteration_tracker[_iter] = temp_timing_iteration["call_tracker"]
				temp_time_tracker[0:temp_timing_iteration["run_times"].shape[0], _iter] = \
					np.sum(temp_timing_iteration["run_times"], axis=1)
				if temp_fitness > temp_best_fitness:
					temp_best_fitness = temp_fitness
				print(f"\tElapsed Time: {temp_elapsed_time:.4f}s\tCurrent Best Fitness: {temp_best_fitness}")
			
			limit_idx = np.max(np.argwhere(temp_fitness_curve_container > 0)[:, 0])
			mimic_results[_size]["RunTime"] = temp_times
			mimic_results[_size]["Curve"] = temp_fitness_curve_container[:limit_idx + 1, :]
			mimic_results[_size]["Fitness"] = temp_fitness_container
			
			mimic_results[_size]["Avg_Iterations"] = np.mean(temp_iteration_tracker)
			mimic_results[_size]["All_Iterations"] = temp_iteration_tracker
			mimic_results[_size]["Alt_runtime"] = temp_time_tracker[:limit_idx, :]
			mimic_results[_size]["Avg_runtime"] = np.mean(mimic_results[_size]["Alt_runtime"], axis=1)
		
		with open(f"{os.getcwd()}/{folder}/All_MIMIC_Results.pkl", "wb") as temp_file:
			pickle.dump(mimic_results, temp_file)
			temp_file.close()
		# endregion
		
		return {"RHC": rhc_results, "SA": sa_results, "GA": ga_results, "MIMIC": mimic_results}
	except Exception as run_evaluations_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'run_evaluations'", run_evaluations_exception)


def read_parameters_from_file(folder, cv=1, size="s", load_file=True, change_pop=False, reset_pop=True):
	try:
		rhc_restarts = np.zeros(shape=(cv,))
		rhc_attempts = np.zeros(shape=(cv,))
		rhc_iters = np.zeros(shape=(cv,))
		
		sa_decay = np.zeros(shape=(cv, cv), dtype=object)
		sa_attempts = np.zeros(shape=(cv,))
		sa_iters = np.zeros(shape=(cv,))
		
		ga_pop_size = np.zeros(shape=(cv,))
		ga_mut_rate = np.zeros(shape=(cv,))
		ga_iters = np.zeros(shape=(cv,))
		ga_attempts = np.zeros(shape=(cv,))
		
		mimic_keep_percent = np.zeros(shape=(cv,))
		mimic_pop_size = np.zeros(shape=(cv,))
		mimic_iters = np.zeros(shape=(cv,))
		mimic_attempts = np.zeros(shape=(cv,))
		
		_directory = f"{os.getcwd()}/{folder}"
		
		csv_files = glob.glob(f"{_directory}/*_{size}_{size}_{size}.csv")
		pkl_files = glob.glob(f"{_directory}/*_{size}_{size}_{size}.pkl")
		# file_names = set([i.split("\\")[-1]
		#                  .split(".")[0] for i in files])
		# alt_file_names = set([i.split("/")[-1].split(".")[0] for i in files])
		if not load_file:
			for fname in csv_files:
				lower_fname = fname.lower()
				if "randomhillclimb" in lower_fname:
					if "restarts" in lower_fname:
						rhc_restarts = np.loadtxt(fname=fname, delimiter=",")
					elif "iteration" in lower_fname:
						rhc_iters = np.loadtxt(fname=fname, delimiter=",")
					elif "attempts" in lower_fname:
						rhc_attempts = np.loadtxt(fname=fname, delimiter=",")
				if "simulatedannealing" in lower_fname:
					if "iteration" in lower_fname:
						sa_iters = np.loadtxt(fname=fname, delimiter=",")
					elif "attempts" in lower_fname:
						sa_attempts = np.loadtxt(fname=fname, delimiter=",")
				if "geneticalgorithm" in lower_fname:
					if "iteration" in lower_fname:
						ga_iters = np.loadtxt(fname=fname, delimiter=",")
					elif "mut_rate" in lower_fname:
						ga_mut_rate = np.loadtxt(fname=fname, delimiter=",")
					elif "pop_size" in lower_fname:
						ga_pop_size = np.loadtxt(fname=fname, delimiter=",")
					elif "attempts" in lower_fname:
						ga_attempts = np.loadtxt(fname=fname, delimiter=",")
				if "mimic" in lower_fname:
					if "iteration" in lower_fname:
						mimic_iters = np.loadtxt(fname=fname, delimiter=",")
					elif "keep_percent" in lower_fname:
						mimic_keep_percent = np.loadtxt(fname=fname, delimiter=",")
					elif "pop_size" in lower_fname:
						mimic_pop_size = np.loadtxt(fname=fname, delimiter=",")
					elif "attempts" in lower_fname:
						mimic_attempts = np.loadtxt(fname=fname, delimiter=",")
			for pklname in pkl_files:
				lower_pkl_name = pklname.lower()
				if "final" in lower_pkl_name:
					continue
				else:
					if "simulatedannealing" in lower_pkl_name:
						sa_decay = np.load(lower_pkl_name, allow_pickle=True)
						if sa_decay.shape[0] > 1:
							sa_decay = sa_decay[np.argmax(sa_decay[:, 1]), 0]
			
			rhc_params = {"restarts": int(np.round(rhc_restarts.mean())),
			              "max_iters": int(np.round(rhc_iters.mean())),
			              "max_attempts": int(np.round(rhc_attempts.mean()))}
			sa_params = {"schedule": sa_decay,
			             "max_iters": int(np.round(sa_iters.mean())),
			             "max_attempts": int(np.round(sa_attempts.mean()))}
			ga_params = {"pop_size": int(np.round((ga_pop_size.mean()))),
			             "mutation_prob": np.round(ga_mut_rate.mean(), 2),
			             "max_attempts": int(np.round(ga_attempts.mean())),
			             "max_iters": int(np.round(mimic_iters.mean()))}
			mimic_params = {"pop_size": int(np.round(mimic_pop_size.mean(), 2)),
			                "keep_pct": np.round(mimic_keep_percent.mean(), 2),
			                "max_attempts": int(np.round(mimic_attempts.mean())),
			                "max_iters": int(np.round(mimic_iters.mean()))}
		else:
			for pklname in pkl_files:
				lower_pkl_name = pklname.lower()
				if "final" in lower_pkl_name:
					if "randomhillclimb" in lower_pkl_name:
						with open(pklname, "rb") as f:
							rhc_params = pickle.load(f)
							f.close()
					elif "simulatedannealing" in lower_pkl_name:
						with open(pklname, "rb") as f:
							sa_params = pickle.load(f)
							f.close()
					elif "geneticalgorithm" in lower_pkl_name:
						with open(pklname, "rb") as f:
							ga_params = pickle.load(f)
							f.close()
					elif "mimic" in lower_pkl_name:
						with open(pklname, "rb") as f:
							mimic_params = pickle.load(f)
							f.close()
		alg_names = ["MIMIC"]
		read_in_parameters = {"RHC": rhc_params, "SA": sa_params, "GA": ga_params, "MIMIC": mimic_params}
		if change_pop and not reset_pop:
			for name in alg_names:
				if read_in_parameters[name]["pop_size"] <= 200:
					read_in_parameters[name]["pop_size"] += 100
					if name == "MIMIC":
						with open(f"{folder}/MIMIC_FINAL_Parameters_{size}_{size}_{size}.pkl", "wb") as f:
							pickle.dump(read_in_parameters[name], f)
							f.close()
		if reset_pop:
			for name in alg_names:
				if name == "MIMIC":
					a = np.loadtxt(f"{_directory}/MIMIC_Best_Pop_Size_{size}_{size}_{size}.csv")
					read_in_parameters[name]["pop_size"] = np.mean(a)
					with open(f"{folder}/MIMIC_FINAL_Parameters_{size}_{size}_{size}.pkl", "wb") as f:
						pickle.dump(read_in_parameters[name], f)
						f.close()
		
		return read_in_parameters
	except Exception as read_parameters_from_file_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in read_parameters_from_file", read_parameters_from_file_exception)


def run_optimization_tests(parameters, prob_name="TSP", size="l", iterations=np.arange(1, 10, 1), verbose=False,
                           maximize=False, gridsearch=True, gen_curves=True, eval_sizes=["s", "m", "l"],
                           max_attempts=500, cv=1, change_pop=False, reset_pop=True, only_size=False):
	try:
		best_parameters = 0
		problem, folder = determine_problem(prob_name=prob_name, size=size, maximize=maximize, SEED=42)
		
		# region Find Optimal Parameters
		sizes = ["s", "m", "l"]
		if only_size:
			sizes = [size]
		if gridsearch:
			for sz in sizes:
				best_rhc_fitness = 0
				best_rhc_params = None
				best_sa_fitness = 0
				best_sa_params = None
				best_ga_fitness = 0
				best_ga_params = None
				best_mimic_fitness = 0
				best_mimic_params = None
				for attempt in max_attempts:
					best_parameters, fits = find_best_parameters(prob_name=prob_name, cv=cv, maximize=maximize,
					                                             max_attempts=attempt,
					                                             size=sz, iterations=iterations, gen_curves=gen_curves,
					                                             parameters=parameters)
					if fits["RHC"] > best_rhc_fitness:
						best_rhc_fitness = fits["RHC"]
						best_rhc_params = best_parameters["RHC"]
					if fits["SA"] > best_sa_fitness:
						best_sa_fitness = fits["SA"]
						best_sa_params = best_parameters["SA"]
					if fits["GA"] > best_ga_fitness:
						best_ga_fitness = fits["GA"]
						best_ga_params = best_parameters["GA"]
					if fits["MIMIC"] > best_mimic_fitness:
						best_mimic_fitness = fits["MIMIC"]
						best_mimic_params = best_parameters["MIMIC"]
				
				with open(f"{folder}/RandomHillClimb_FINAL_Parameters_{sz}_{sz}_{sz}.pkl", "wb") as f:
					pickle.dump(best_rhc_params, f)
					f.close()
				with open(f"{folder}/SimulatedAnnealing_FINAL_Parameters_{sz}_{sz}_{sz}.pkl", "wb") as f:
					pickle.dump(best_sa_params, f)
					f.close()
				with open(f"{folder}/GeneticAlgorithm_FINAL_Parameters_{sz}_{sz}_{sz}.pkl", "wb") as f:
					pickle.dump(best_ga_params, f)
					f.close()
				with open(f"{folder}/MIMIC_FINAL_Parameters_{sz}_{sz}_{sz}.pkl", "wb") as f:
					pickle.dump(best_mimic_params, f)
					f.close()
		# endregion
		
		# region Read in stored parameters
		if not gridsearch:
			# Read in best parameters
			best_parameters = read_parameters_from_file(folder=folder, cv=cv, size=size, change_pop=change_pop,
			                                            reset_pop=reset_pop)
		# endregion
		
		# region Problem Evaluation of Fitness and Runtimes
		if prob_name != "NN" or prob_name != "nn":
			print(f"Starting Discrete Problem Evaluation")
		else:
			print(f"Starting Neural Network Problem Evaluation")
		if only_size:
			run_evaluations(prob_name=prob_name, params=best_parameters, eval_sizes=eval_sizes,
			                max_attempts=max_attempts, change_pop=change_pop, reset_pop=reset_pop)
		else:
			eval_sizes = sizes
		
		# endregion
		return
	
	except Exception as run_optimization_tests_except:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'run_optimization_tests'.\n", run_optimization_tests_except)
		print()


def check_folder(_directory):
	MYDIR = os.getcwd() + "\\" + _directory
	CHECK_FOLDER = os.path.isdir(MYDIR)
	
	# If folder doesn't exist, then create it.
	if not CHECK_FOLDER:
		os.makedirs(MYDIR)
		print("created folder : ", MYDIR)
	else:
		print(MYDIR, "folder already exists.")


def plot_between(avg, std, x, cm, colors, xlabel, ylabel, title, algorithm_name, folder, size, ax=None,
                 extra_name=None, use_log_x=False, use_log_y=False):
	try:
		if ax is None:
			plt.close("all")
			plt.grid()
			plt.grid(which='major', linestyle='-', linewidth='0.5', color='white')
			plt.fill_between(x, avg - std, avg + std, alpha=0.2)
			if extra_name is not None:
				if use_log_y:
					plt.semilogy(x, avg, 'o-', label=f"{algorithm_name} - {extra_name}", markersize=4)
				else:
					plt.plot(x, avg, 'o-', label=f"{algorithm_name} - {extra_name}", markersize=4)
			else:
				if use_log_y:
					plt.semilogy(x, avg, 'o-', label=f"{algorithm_name}", markersize=4)
				else:
					plt.plot(x, avg, 'o-', label=f"{algorithm_name}", markersize=4)
			
			plt.title(title, fontsize=15, weight='bold')
			plt.ylim((0, avg.max() + (avg.max() * 0.1)))
			plt.xlabel(xlabel=xlabel, fontsize=15, weight='heavy')
			plt.ylabel(ylabel=ylabel, fontsize=15, weight='heavy')
			plt.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
			plt.tight_layout()
			if extra_name is not None:
				plt.savefig(f"{os.getcwd()}/{folder}/{folder}_{algorithm_name}_{size}_Runtime.png", bbox_inches='tight')
			else:
				plt.savefig(f"{os.getcwd()}/{folder}/{folder}_{algorithm_name}_{size}_Fitness.png", bbox_inches='tight')
			plt.close('all')
			return
		else:
			# Not using a color so matplotlib will auto assign
			ax.fill_between(x, avg - std, avg + std, alpha=0.25)
			
			# Making marker size smaller since we have multiple lines on graph
			if use_log_y:
				ax.semilogy(x, avg, 'o-', label=f"{algorithm_name}{extra_name}", markersize=2)
			else:
				ax.plot(x, avg, 'o-', label=f"{algorithm_name}{extra_name}", markersize=2)
			return ax
	except Exception as plot_between_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'plot_between'.", plot_between_exception)


def find_array_expansion(data, key, size=None):
	try:
		if size is not None:
			tmp = np.zeros(shape=(4,))
			tmp[0] = data["RHC"][size][key].shape[0]
			tmp[1] = data["SA"][size][key].shape[0]
			tmp[2] = data["GA"][size][key].shape[0]
			tmp[3] = data["MIMIC"][size][key].shape[0]
			return int(np.round(np.max(tmp) * 1.1))
		else:
			tmp = np.zeros(shape=(3,))
			tmp[0] = data["l"][key].shape[0]
			tmp[1] = data["m"][key].shape[0]
			tmp[2] = data["s"][key].shape[0]
			return int(np.round(np.max(tmp) * 1.1))
	except Exception as find_array_expansion_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'find_array_expansion'.", find_array_expansion_exception)


def plot_discrete_compare_size(results, folder, prob_name, alg_name):
	try:
		# Compare the different sizes
		cm = plt.get_cmap('tab20c')
		colors = [cm(1. * i / 10) for i in range(10)]
		
		# region Large Problem
		temp_array = np.zeros(
			shape=(find_array_expansion(results, "Alt_runtime"), results["l"]["Alt_runtime"].shape[1]))
		temp_array[:results["l"]["Alt_runtime"].shape[0], :results["l"]["Alt_runtime"].shape[1]] = results["l"][
			"Alt_runtime"]
		run_times_large = pd.DataFrame(temp_array)
		run_times_large.replace(0.0, np.nan, inplace=True)
		run_times_large.fillna(method='ffill', inplace=True)
		run_times_large.fillna(method='bfill', inplace=True)
		avg_runtime_per_iteration_large = run_times_large.sum()
		avg_runtime_overall_large = np.round(np.mean(avg_runtime_per_iteration_large), 2)
		temp_avg = run_times_large.mean(axis=1)
		temp_std = run_times_large.std(axis=1)
		run_times_large["AVG"] = temp_avg
		run_times_large["STD"] = temp_std
		plot_between(avg=run_times_large["AVG"], std=run_times_large["STD"],
		             x=run_times_large.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(large)",
		             ylabel="Runtime(s)", xlabel="Iterations", size="large", algorithm_name=alg_name, folder=folder,
		             extra_name=f" - AVG {avg_runtime_overall_large:.4f}s")
		
		temp_array = np.zeros(shape=(find_array_expansion(results, "Curve"), results["l"]["Curve"].shape[1]))
		temp_array[:results["l"]["Curve"].shape[0], :results["l"]["Curve"].shape[1]] = results["l"]["Curve"]
		large_fitness_df = pd.DataFrame(data=temp_array)
		large_fitness_df.replace(0.0, np.nan, inplace=True)
		large_fitness_df.fillna(method="ffill", inplace=True)
		large_fitness_df.fillna(method="bfill", inplace=True)
		temp_avg = large_fitness_df.mean(axis=1)
		temp_std = large_fitness_df.std(axis=1)
		large_fitness_df["AVG"] = temp_avg
		large_fitness_df["STD"] = temp_std
		plot_between(avg=large_fitness_df["AVG"], std=large_fitness_df["STD"],
		             x=large_fitness_df.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(large)",
		             ylabel="Fitness", xlabel="Iterations", size="large", algorithm_name=alg_name, folder=folder,
		             extra_name=f"Fitness: {large_fitness_df['AVG'].iloc[-1]}")
		# endregion
		
		# region Medium Problem
		temp_array = np.zeros(shape=(find_array_expansion(results, "Alt_runtime"),
		                             results["l"]["Alt_runtime"].shape[1]))
		temp_array[:results["m"]["Alt_runtime"].shape[0], :results["m"]["Alt_runtime"].shape[1]] = \
			results["m"]["Alt_runtime"]
		run_times_medium = pd.DataFrame(temp_array)
		run_times_medium.replace(0.0, np.nan, inplace=True)
		run_times_medium.fillna(method='ffill', inplace=True)
		run_times_medium.fillna(method='bfill', inplace=True)
		avg_runtime_per_iteration_medium = run_times_medium.sum()
		avg_runtime_overall_medium = np.round(np.mean(avg_runtime_per_iteration_medium), 2)
		temp_avg = run_times_medium.mean(axis=1)
		temp_std = run_times_medium.std(axis=1)
		run_times_medium["AVG"] = temp_avg
		run_times_medium["STD"] = temp_std
		plot_between(avg=run_times_medium["AVG"], std=run_times_medium["STD"],
		             x=run_times_medium.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(medium)",
		             ylabel="Runtime(s)", xlabel="Iterations", size="medium", algorithm_name=alg_name, folder=folder,
		             extra_name=f" - AVG {avg_runtime_overall_medium:.4f}s")
		
		temp_array = np.zeros(shape=(find_array_expansion(results, "Curve"),
		                             results["l"]["Curve"].shape[1]))
		temp_array[:results["m"]["Curve"].shape[0], :results["m"]["Curve"].shape[1]] = results["m"]["Curve"]
		medium_df = pd.DataFrame(data=temp_array)
		medium_df.replace(0.0, np.nan, inplace=True)
		medium_df.fillna(method="ffill", inplace=True)
		medium_df.fillna(method="bfill", inplace=True)
		temp_avg = medium_df.mean(axis=1)
		temp_std = medium_df.std(axis=1)
		medium_df["AVG"] = temp_avg
		medium_df["STD"] = temp_std
		plot_between(avg=medium_df["AVG"], std=medium_df["STD"],
		             x=medium_df.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(medium)",
		             ylabel="Fitness", xlabel="Iterations", size="medium", algorithm_name=alg_name, folder=folder,
		             extra_name=f"Fitness: {medium_df['AVG'].iloc[-1]}")
		# endregion
		
		# region Small Problem
		temp_array = np.zeros(shape=(find_array_expansion(results, "Alt_runtime"),
		                             results["l"]["Alt_runtime"].shape[1]))
		temp_array[:results["s"]["Alt_runtime"].shape[0], :results["s"]["Alt_runtime"].shape[1]] = \
			results["s"]["Alt_runtime"]
		run_times_small = pd.DataFrame(temp_array)
		run_times_medium.replace(0.0, np.nan, inplace=True)
		run_times_medium.fillna(method='ffill', inplace=True)
		run_times_medium.fillna(method='bfill', inplace=True)
		avg_runtime_per_iteration_small = run_times_small.sum()
		avg_runtime_overall_small = np.round(np.mean(avg_runtime_per_iteration_small), 2)
		temp_avg = run_times_small.mean(axis=1)
		temp_std = run_times_small.std(axis=1)
		run_times_small["AVG"] = temp_avg
		run_times_small["STD"] = temp_std
		plot_between(avg=run_times_small["AVG"], std=run_times_small["STD"],
		             x=run_times_small.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(small)",
		             ylabel="Runtime(s)", xlabel="Iterations", size="small", algorithm_name=alg_name, folder=folder,
		             extra_name=f" - AVG {avg_runtime_overall_small:.4f}s")
		
		temp_array = np.zeros(shape=(find_array_expansion(results, "Curve"), results["l"]["Curve"].shape[1]))
		temp_array[:results["s"]["Curve"].shape[0], :results["s"]["Curve"].shape[1]] = results["s"]["Curve"]
		small_df = pd.DataFrame(data=temp_array)
		small_df.replace(0.0, np.nan, inplace=True)
		small_df.fillna(method="ffill", inplace=True)
		small_df.fillna(method="bfill", inplace=True)
		temp_avg = small_df.mean(axis=1)
		temp_std = small_df.std(axis=1)
		small_df["AVG"] = temp_avg
		small_df["STD"] = temp_std
		plot_between(avg=small_df["AVG"], std=small_df["STD"],
		             x=small_df.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(small)",
		             ylabel="Fitness", xlabel="Iterations", size="small", algorithm_name=alg_name, folder=folder,
		             extra_name=f"Fitness: {small_df['AVG'].iloc[-1]}")
		# endregion
		
		# region Plot Fitness all sizes
		plt.close("all")
		fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))
		
		plot_between(avg=large_fitness_df["AVG"], std=large_fitness_df["STD"],
		             x=large_fitness_df.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(large)",
		             ylabel="Fitness", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax1, extra_name=f" Large - Fitness: {large_fitness_df['AVG'].iloc[-1]}")
		
		plot_between(avg=medium_df["AVG"], std=medium_df["STD"],
		             x=medium_df.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(medium)",
		             ylabel="Fitness", xlabel="Iterations", size="medium", algorithm_name=alg_name,
		             folder=folder, ax=ax1, extra_name=f" Medium - Fitness: {medium_df['AVG'].iloc[-1]}")
		
		plot_between(avg=small_df["AVG"], std=small_df["STD"],
		             x=small_df.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(small)",
		             ylabel="Fitness", xlabel="Iterations", size="small", algorithm_name=alg_name,
		             folder=folder, ax=ax1, extra_name=f" Small - Fitness: {small_df['AVG'].iloc[-1]}")
		
		ax1.set_prop_cycle(cycler('color', colors))
		ax1.set_title(f"{prob_name}\n {alg_name} Fitness", fontsize=15, weight='bold')
		ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
		ax1.set_xlabel("Iterations", fontsize=15, weight='heavy')
		ax1.set_ylabel("Fitness", fontsize=15, weight='heavy')
		ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
		plt.tight_layout()
		plt.savefig(f"{os.getcwd()}/{folder}/{folder}_{alg_name}_Combined_sizes_Fitness.png", bbox_inches='tight')
		plt.close("all")
		# endregion
		
		# region Plot Runtimes all sizes
		plt.close("all")
		fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))
		
		plot_between(avg=run_times_large["AVG"], std=run_times_large["STD"],
		             x=run_times_large.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(large)",
		             ylabel="Runtime(s)", xlabel="Iterations", size="large", algorithm_name=alg_name, folder=folder,
		             ax=ax1, extra_name=f" Large - AVG: {avg_runtime_overall_large:.4f}s")
		
		plot_between(avg=run_times_medium["AVG"], std=run_times_medium["STD"],
		             x=run_times_medium.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(medium)",
		             ylabel="Runtime(s)", xlabel="Iterations", size="medium", algorithm_name=alg_name, folder=folder,
		             ax=ax1, extra_name=f" Medium - AVG: {avg_runtime_overall_medium:.4f}s")
		
		plot_between(avg=run_times_small["AVG"], std=run_times_small["STD"],
		             x=run_times_small.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(small)",
		             ylabel="Runtime(s)", xlabel="Iterations", size="small", algorithm_name=alg_name, folder=folder,
		             ax=ax1, extra_name=f" Small - AVG: {avg_runtime_overall_small:.4f}s")
		
		ax1.set_prop_cycle(cycler('color', colors))
		ax1.set_title(f"{prob_name}\n {alg_name} Runtimes", fontsize=15, weight='bold')
		ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
		ax1.set_xlabel("Iterations", fontsize=15, weight='heavy')
		ax1.set_ylabel("Runtime(s)", fontsize=15, weight='heavy')
		ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
		plt.tight_layout()
		plt.savefig(f"{os.getcwd()}/{folder}/{folder}_{alg_name}_Combined_sizes_Runtime.png", bbox_inches='tight')
		plt.close("all")
		# endregion
		
		# Plot Fitness and runtimes with varying sizes
		
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
		
		plot_between(avg=large_fitness_df["AVG"], std=large_fitness_df["STD"],
		             x=large_fitness_df.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(large)",
		             ylabel="Fitness", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax1, extra_name=f" Large - Fitness: {large_fitness_df['AVG'].iloc[-1]}")
		
		plot_between(avg=medium_df["AVG"], std=medium_df["STD"],
		             x=medium_df.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(medium)",
		             ylabel="Fitness", xlabel="Iterations", size="medium", algorithm_name=alg_name,
		             folder=folder, ax=ax1, extra_name=f" Medium - Fitness: {medium_df['AVG'].iloc[-1]}")
		
		plot_between(avg=small_df["AVG"], std=small_df["STD"],
		             x=small_df.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(small)",
		             ylabel="Fitness", xlabel="Iterations", size="small", algorithm_name=alg_name,
		             folder=folder, ax=ax1, extra_name=f" Small - Fitness: {small_df['AVG'].iloc[-1]}")
		
		ax1.set_prop_cycle(cycler('color', colors))
		ax1.set_title(f"{prob_name}\n {alg_name} Fitness", fontsize=15, weight='bold')
		ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
		ax1.set_xlabel("Iterations", fontsize=15, weight='heavy')
		ax1.set_ylabel("Fitness", fontsize=15, weight='heavy')
		ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
		
		plot_between(avg=run_times_large["AVG"], std=run_times_large["STD"],
		             x=run_times_large.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(large)",
		             ylabel="Runtime(s)", xlabel="Iterations", size="large", algorithm_name=alg_name, folder=folder,
		             ax=ax2, extra_name=f" Large - AVG: {avg_runtime_overall_large:.4f}s")
		
		plot_between(avg=run_times_medium["AVG"], std=run_times_medium["STD"],
		             x=run_times_medium.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(medium)",
		             ylabel="Runtime(s)", xlabel="Iterations", size="medium", algorithm_name=alg_name, folder=folder,
		             ax=ax2, extra_name=f" Medium - AVG: {avg_runtime_overall_medium:.4f}s")
		
		plot_between(avg=run_times_small["AVG"], std=run_times_small["STD"],
		             x=run_times_small.index.values, cm=cm, colors=colors, title=f"{prob_name} - {alg_name}(small)",
		             ylabel="Runtime(s)", xlabel="Iterations", size="small", algorithm_name=alg_name, folder=folder,
		             ax=ax2, extra_name=f" Small - AVG: {avg_runtime_overall_small:.4f}s")
		
		ax2.set_prop_cycle(cycler('color', colors))
		ax2.set_title(f"{prob_name}\n {alg_name} Runtimes", fontsize=15, weight='bold')
		ax2.grid(which='major', linestyle='-', linewidth='0.5', color='white')
		ax2.set_xlabel("Iterations", fontsize=15, weight='heavy')
		ax2.set_ylabel("Runtime", fontsize=15, weight='heavy')
		ax2.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
		plt.tight_layout()
		plt.savefig(f"{os.getcwd()}/{folder}/{folder}_{alg_name}_Combined_sizes_Fitness_and_Runtime.png",
		            bbox_inches='tight')
		plt.close('all')
		return
	
	except Exception as plot_discrete_compare_size_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'plot_discrete_compare_size'.", plot_discrete_compare_size_exception)


def load_all_results_from_file(folder, names=["RHC"]):
	try:
		loaded_data = {}
		for alg_name in names:
			alg_name = alg_name.upper()
			with open(f"{os.getcwd()}/{folder}/All_{alg_name}_Results.pkl", "rb") as input_file:
				loaded_data[f"{alg_name}"] = pickle.load(input_file)
				input_file.close()
		
		return loaded_data
	except Exception as load_all_data_from_file_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'load_all_data_from_file'.", load_all_data_from_file_exception)


def add_avg_and_std(data, rows, columns):
	try:
		temp_array = np.zeros(shape=(rows, data.shape[1]))
		temp_array[:data.shape[0], :data.shape[1]] = data
		temp_df = pd.DataFrame(temp_array)
		temp_df.replace(0.0, np.nan, inplace=True)
		temp_df.fillna(method="ffill", inplace=True)
		temp_df.fillna(method="bfill", inplace=True)
		avg = temp_df.mean(axis=1)
		std = temp_df.std(axis=1)
		temp_df["AVG"] = avg
		temp_df["STD"] = std
		return temp_df
	except Exception as add_avg_and_std_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'add_avg_and_std'.", add_avg_and_std_exception)


def plot_discrete_compare_algorithm(results, folder, prob_name):
	try:
		cm = plt.get_cmap('tab20c')
		colors = [cm(1. * i / 10) for i in range(10)]
		all_data = load_all_results_from_file(folder=folder, names=["RHC", "GA", "SA", "MIMIC"])
		
		# Plot Fitness all algorithms
		lim = find_array_expansion(all_data, "Curve", size="l")
		rhc_fitness = add_avg_and_std(all_data["RHC"]["l"]["Curve"], lim, None)
		sa_fitness = add_avg_and_std(all_data["SA"]["l"]["Curve"], lim, None)
		ga_fitness = add_avg_and_std(all_data["GA"]["l"]["Curve"], lim, None)
		mimic_fitness = add_avg_and_std(all_data["MIMIC"]["l"]["Curve"], lim, None)
		
		plt.close("all")
		fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))
		# RHC Fitness
		alg_name = "RHC"
		plot_between(avg=rhc_fitness["AVG"], std=rhc_fitness["STD"],
		             x=rhc_fitness["AVG"].index.values, cm=cm, colors=colors, title=f"{prob_name} - Fitness",
		             ylabel="Fitness", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax1, extra_name=f" - Fitness: {rhc_fitness['AVG'].iloc[-1]}")
		# SA Fitness
		alg_name = "SA"
		plot_between(avg=sa_fitness["AVG"], std=sa_fitness["STD"],
		             x=sa_fitness["AVG"].index.values, cm=cm, colors=colors,
		             title=f"{prob_name} - Fitness",
		             ylabel="Fitness", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax1, extra_name=f" - Fitness: {sa_fitness['AVG'].iloc[-1]}")
		# GA Fitness
		alg_name = "GA"
		plot_between(avg=ga_fitness["AVG"], std=ga_fitness["STD"],
		             x=ga_fitness["AVG"].index.values, cm=cm, colors=colors,
		             title=f"{prob_name} - Fitness",
		             ylabel="Fitness", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax1, extra_name=f" - Fitness: {ga_fitness['AVG'].iloc[-1]}")
		# MIMIC Fitness
		alg_name = "MIMIC"
		plot_between(avg=mimic_fitness["AVG"], std=mimic_fitness["STD"],
		             x=mimic_fitness["AVG"].index.values, cm=cm, colors=colors,
		             title=f"{prob_name} - Fitness",
		             ylabel="Fitness", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax1, extra_name=f" - Fitness: {mimic_fitness['AVG'].iloc[-1]}")
		
		ax1.set_prop_cycle(cycler('color', colors))
		ax1.set_title(f"{prob_name} Fitness\n All Algorithms", fontsize=15, weight='bold')
		ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
		ax1.set_xlabel("Iterations", fontsize=15, weight='heavy')
		ax1.set_ylabel("Fitness", fontsize=15, weight='heavy')
		ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
		plt.tight_layout()
		plt.savefig(f"{os.getcwd()}/{folder}/{folder}_Combined_sizes_Fitness_All_Algorithms.png",
		            bbox_inches='tight')
		plt.close("all")
		
		# Plot Runtimes all algorithms
		plt.close("all")
		fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))
		lim = find_array_expansion(all_data, "Alt_runtime", size="l")
		rhc_runtime = add_avg_and_std(all_data["RHC"]["l"]["Alt_runtime"], lim, None)
		sa_runtime = add_avg_and_std(all_data["SA"]["l"]["Alt_runtime"], lim, None)
		ga_runtime = add_avg_and_std(all_data["GA"]["l"]["Alt_runtime"], lim, None)
		mimic_runtime = add_avg_and_std(all_data["MIMIC"]["l"]["Alt_runtime"], lim, None)
		
		# RHC Runtime
		alg_name = "RHC"
		plot_between(avg=rhc_runtime["AVG"], std=rhc_runtime["STD"],
		             x=rhc_runtime["AVG"].index.values, cm=cm, colors=colors,
		             title=f"{prob_name} - Runtimes",
		             ylabel="Runtime(s)", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax1, use_log_y=True, extra_name=f" - AVG: {rhc_runtime['AVG'].mean():.4f}s")
		# SA Runtime
		alg_name = "SA"
		plot_between(avg=sa_runtime["AVG"], std=sa_runtime["STD"],
		             x=sa_runtime["AVG"].index.values, cm=cm, colors=colors,
		             title=f"{prob_name} - Runtimes",
		             ylabel="Runtime(s)", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax1, use_log_y=True, extra_name=f" - AVG: {sa_runtime['AVG'].mean():.4f}s")
		# GA Runtime
		alg_name = "GA"
		plot_between(avg=ga_runtime["AVG"], std=ga_runtime["STD"],
		             x=ga_runtime["AVG"].index.values, cm=cm, colors=colors,
		             title=f"{prob_name} - Runtimes",
		             ylabel="Runtime(s)", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax1, use_log_y=True, extra_name=f" - AVG: {ga_runtime['AVG'].mean():.4f}s")
		# MIMIC Runtime
		alg_name = "MIMIC"
		plot_between(avg=mimic_runtime["AVG"], std=mimic_runtime["STD"],
		             x=mimic_runtime["AVG"].index.values, cm=cm, colors=colors,
		             title=f"{prob_name} - Runtimes",
		             ylabel="Runtime(s)", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax1, use_log_y=True, extra_name=f" - AVG: {mimic_runtime['AVG'].mean():.4f}s")
		
		ax1.set_prop_cycle(cycler('color', colors))
		ax1.set_title(f"{prob_name} Runtimes\n All Algorithms", fontsize=15, weight='bold')
		ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
		ax1.set_xlabel("Iterations", fontsize=15, weight='heavy')
		ax1.set_ylabel("Runtimes(s)", fontsize=15, weight='heavy')
		ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
		plt.tight_layout()
		plt.savefig(f"{os.getcwd()}/{folder}/{folder}_Combined_sizes_Runtimes_All_Algorithms.png",
		            bbox_inches='tight')
		plt.close("all")
		
		# Plot Combined Fitness and Runtimes all algorithms
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
		
		# RHC Fitness
		alg_name = "RHC"
		plot_between(avg=rhc_fitness["AVG"], std=rhc_fitness["STD"],
		             x=rhc_fitness["AVG"].index.values, cm=cm, colors=colors,
		             title=f"{prob_name} - Fitness",
		             ylabel="Fitness", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax1, extra_name=f" - Fitness: {rhc_fitness['AVG'].iloc[-1]}")
		# SA Fitness
		alg_name = "SA"
		plot_between(avg=sa_fitness["AVG"], std=sa_fitness["STD"],
		             x=sa_fitness["AVG"].index.values, cm=cm, colors=colors,
		             title=f"{prob_name} - Fitness",
		             ylabel="Fitness", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax1, extra_name=f" - Fitness: {sa_fitness['AVG'].iloc[-1]}")
		# GA Fitness
		alg_name = "GA"
		plot_between(avg=ga_fitness["AVG"], std=ga_fitness["STD"],
		             x=ga_fitness["AVG"].index.values, cm=cm, colors=colors,
		             title=f"{prob_name} - Fitness",
		             ylabel="Fitness", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax1, extra_name=f" - Fitness: {ga_fitness['AVG'].iloc[-1]}")
		# MIMIC Fitness
		alg_name = "MIMIC"
		plot_between(avg=mimic_fitness["AVG"], std=mimic_fitness["STD"],
		             x=mimic_fitness["AVG"].index.values, cm=cm, colors=colors,
		             title=f"{prob_name} - Fitness",
		             ylabel="Fitness", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax1, extra_name=f" - Fitness: {mimic_fitness['AVG'].iloc[-1]}")
		
		ax1.set_prop_cycle(cycler('color', colors))
		ax1.set_title(f"{prob_name} Fitness\n All Algorithms", fontsize=15, weight='bold')
		ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
		ax1.set_xlabel("Iterations", fontsize=15, weight='heavy')
		ax1.set_ylabel("Fitness", fontsize=15, weight='heavy')
		ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
		
		# RHC Runtime
		alg_name = "RHC"
		plot_between(avg=rhc_runtime["AVG"], std=rhc_runtime["STD"],
		             x=rhc_runtime["AVG"].index.values, cm=cm, colors=colors,
		             title=f"{prob_name} - Runtime",
		             ylabel="Runtime(s)", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax2, use_log_y=True, extra_name=f" - AVG: {rhc_runtime['AVG'].mean():.4f}s")
		# SA Runtime
		alg_name = "SA"
		plot_between(avg=sa_runtime["AVG"], std=sa_runtime["STD"],
		             x=sa_runtime["AVG"].index.values, cm=cm, colors=colors,
		             title=f"{prob_name} - Runtime",
		             ylabel="Runtime(s)", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax2, use_log_y=True, extra_name=f" - AVG: {sa_runtime['AVG'].mean():.4f}s")
		# GA Runtime
		alg_name = "GA"
		plot_between(avg=ga_runtime["AVG"], std=ga_runtime["STD"],
		             x=ga_runtime["AVG"].index.values, cm=cm, colors=colors,
		             title=f"{prob_name} - Runtime",
		             ylabel="Runtime(s)", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax2, use_log_y=True, extra_name=f" - AVG: {ga_runtime['AVG'].mean():.4f}s")
		# MIMIC Runtime
		alg_name = "MIMIC"
		plot_between(avg=mimic_runtime["AVG"], std=mimic_runtime["STD"],
		             x=mimic_runtime["AVG"].index.values, cm=cm, colors=colors,
		             title=f"{prob_name} - Runtime",
		             ylabel="Runtime(s)", xlabel="Iterations", size="large", algorithm_name=alg_name,
		             folder=folder, ax=ax2, use_log_y=True, extra_name=f" - AVG: {mimic_runtime['AVG'].mean():.4f}s")
		
		ax2.set_prop_cycle(cycler('color', colors))
		ax2.set_title(f"{prob_name} Runtimes\n All Algorithms", fontsize=15, weight='bold')
		ax2.grid(which='major', linestyle='-', linewidth='0.5', color='white')
		ax2.set_xlabel("Iterations", fontsize=15, weight='heavy')
		ax2.set_ylabel("Runtime", fontsize=15, weight='heavy')
		ax2.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
		plt.tight_layout()
		plt.savefig(f"{os.getcwd()}/{folder}/{folder}_Combined_sizes_Fitness_and_Runtime_All_Algorithms.png",
		            bbox_inches='tight')
		plt.close('all')
	
	except Exception as plot_discrete_compare_algorithm_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(f"Exception in 'plot_discrete_compare_algorithm'.", plot_discrete_compare_algorithm_exception)


def plot_discrete(all_results, folder, prob_name, alg_name):
	try:
		
		# Compare the various algorithms with the different sizes of the problem (s,m,l) with themselves
		plot_discrete_compare_size(results=all_results, folder=folder, prob_name=prob_name, alg_name=alg_name)
		
		# Compare the various algorithms with different sizes of the problem (s,m,l) with each other. large vs large
		plot_discrete_compare_algorithm(results=all_results, folder=folder, prob_name=prob_name)
	
	# Compare runtimes for each
	except Exception as plot_discrete_exception:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print("Exception in 'plot_discrete_exception'.", plot_discrete_exception)


def plot_count(count_df, prob_name, x_label=None, y_label=None, title=None, ax=None, use_log_y=False,
               f_name=f"__2_", title_helper=""):
	plt.close("all")
	cm = plt.get_cmap('tab20c')
	colors = [cm(1. * i / 10) for i in range(10)]
	if ax is not None:
		if use_log_y:
			ax.semilogy(count_df.index, count_df["RHC"], 'o-', label=f"RHC", markersize=6)
			ax.semilogy(count_df.index, count_df["SA"], 'o-', label=f"SA", markersize=6)
			ax.semilogy(count_df.index, count_df["GA"], 'o-', label=f"GA", markersize=6)
			ax.semilogy(count_df.index, count_df["MIMIC"], 'o-', label=f"MIMIC", markersize=6)
		else:
			count_df.plot(ax=ax, marker="o", markersize=6)
		ax.set_prop_cycle(cycler('color', colors))
		ax.set_title(f"{title}", fontsize=15, weight='bold')
		ax.grid(which='major', linestyle='-', linewidth='0.5', color='white')
		ax.set_xlabel(f"{x_label}", fontsize=15, weight='heavy')
		ax.set_ylabel(f"{y_label}", fontsize=15, weight='heavy')
		ax.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
		plt.tight_layout()
		return ax
	else:
		fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))
		if use_log_y:
			ax1.semilogy(count_df.index, count_df["RHC"], 'o-', label=f"RHC", markersize=6)
			ax1.semilogy(count_df.index, count_df["SA"], 'o-', label=f"SA", markersize=6)
			ax1.semilogy(count_df.index, count_df["GA"], 'o-', label=f"GA", markersize=6)
			ax1.semilogy(count_df.index, count_df["MIMIC"], 'o-', label=f"MIMIC", markersize=6)
		else:
			count_df.plot(ax=ax1, marker="o", markersize=6)
		ax1.set_prop_cycle(cycler('color', colors))
		ax1.set_title(f"{title}", fontsize=15, weight='bold')
		ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
		ax1.set_xlabel(f"{x_label}", fontsize=15, weight='heavy')
		ax1.set_ylabel(f"{y_label}", fontsize=15, weight='heavy')
		ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
		plt.tight_layout()
		plt.savefig(f"{os.getcwd()}/{prob_name}/Function_Evaluation_Calls_{f_name}.png", bbox_inches='tight')
		plt.close("all")
		return
	

def empty_func(count_df, times_df, problem_name, folder):
	plt.close("all")
	
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
	
	plot_count(count_df, "FlipFlop", "Problem Size", "Function Evaluations",
	               f"{problem_name}\nFunction Evaluations Vs. Problem Size",
	               use_log_y=False, ax=ax1)
	
	plot_count(times_df, "FlipFlop", "Problem Size", "Runtimes",
	               f"{problem_name}\nRuntime Vs. Problem Size",
	               use_log_y=False, ax=ax2)
	plt.savefig(f"{os.getcwd()}/{folder}/Evaluation_vs_Size_combined.png")
	plt.close("all")
	return


def plot_func(df):
	plt.close("all")
	cm = plt.get_cmap('tab20c')
	colors = [cm(1. * i / 10) for i in range(10)]
	fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))
	df.plot(ax=ax1)
	ax1.set_prop_cycle(cycler('color', colors))
	ax1.set_title(f"Fitness Curves", fontsize=15, weight='bold')
	ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
	ax1.set_xlabel(f"Iterations", fontsize=15, weight='heavy')
	ax1.set_ylabel(f"Fitness", fontsize=15, weight='heavy')
	ax1.legend(loc="best", markerscale=1.1, frameon=True, edgecolor="black")
	plt.tight_layout()
	plt.savefig(f"{os.getcwd()}/Fitness_vs_iterations.png")