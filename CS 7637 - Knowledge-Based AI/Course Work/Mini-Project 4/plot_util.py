import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [12, 8]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500


# noinspection DuplicatedCode
def plot_single(encoded_data, title, xlabel, ylabel, trait_labels, y_limit=8, ax=None,
                color="tab:green", marker="1", label="Positive", type="single", save_file_name=""):
	if ax is None:
		fig, ax = plt.subplots(figsize=(8, 6))
	scatter_data = np.asarray(
		[[j, encoded_data[i][j]] for i in range(len(encoded_data)) for j in range(len(encoded_data[i]))])
	dataframe_for_scatter_plot = pd.DataFrame(data={"x": scatter_data[:, 0], "y": scatter_data[:, 1]})
	dataframe_for_scatter_plot.plot.scatter(x="x", y="y", ax=ax, color=color, marker=marker,
	                                        s=100, label=f"{label}")
	ax.set_title(f"{title}", fontsize=15, weight='bold')
	ax.grid(which='major', linestyle='-', linewidth='0.5', color='white')
	ax.set_xlabel(f"{xlabel}", fontsize=15, weight='heavy')
	ax.set_ylabel(f"{ylabel}", fontsize=15, weight='heavy')
	ax.set_xlim([-1, encoded_data.shape[1]])
	ax.set_xmargin(0.5)
	ax.set_ylim([-1, y_limit + 1])
	ax.set_ymargin(0.5)
	ax.set_xticks(np.arange(encoded_data.shape[1]))
	ax.set_xticklabels(trait_labels)
	ax.set_yticks(np.arange(y_limit + 1))
	plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
	         rotation_mode="anchor")
	if type == "single":
		plt.tight_layout()
		plt.savefig(f"{save_file_name}.png")
	else:
		return ax


def plot_combined(positive_examples_encoded, negative_examples_encoded, title, xlabel, ylabel, trait_labels, y_limit=8,
                  save_name=""):
	fig, ax1 = plt.subplots(figsize=(positive_examples_encoded.shape[1], 6))
	plot_single(encoded_data=positive_examples_encoded, title=title, xlabel=xlabel,
	            ylabel=ylabel, trait_labels=trait_labels, y_limit=y_limit, ax=ax1,
	            color="tab:green", marker="1", label="Positive Instances", type="combined")
	plot_single(encoded_data=negative_examples_encoded, title=title, xlabel=xlabel,
	            ylabel=ylabel, trait_labels=trait_labels, y_limit=y_limit, ax=ax1,
	            color="tab:red", marker="2", label="Negative Instances", type="combined")
	ax1.legend(loc="best", markerscale=1.1, frameon=True,
	           edgecolor="black", fancybox=True, shadow=True, fontsize=12)
	plt.tight_layout()
	plt.savefig(f"{save_name}.png")
