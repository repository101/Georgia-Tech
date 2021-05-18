import copy
import glob
import os
import pickle
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500

plt.tight_layout()


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

