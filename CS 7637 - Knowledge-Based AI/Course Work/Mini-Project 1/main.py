from SemanticNetsAgent import SemanticNetsAgent
import pandas as pd
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500


def test():
    # This will test your SemanticNetsAgent
    # #with seven initial test cases.
    test_agent = SemanticNetsAgent()
    start_time = time.time()
    print(test_agent.solve(10000, 9989))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Runtime for ({test_agent.initial_sheep}, {test_agent.initial_wolves}): {elapsed_time:.5f}s")
    # print(test_agent.solve(11, 10))
    # print(test_agent.solve(51, 50))
    # print(test_agent.solve(101, 100))
    # print(test_agent.solve(201, 200))
    #
    # print(test_agent.solve(16, 15))
    # print(test_agent.solve(1, 1))
    # print(test_agent.solve(2, 2))
    # print(test_agent.solve(3, 3))
    # print(test_agent.solve(5, 3))
    # print(test_agent.solve(6, 3))
    print(test_agent.solve(7, 3))
    test_agent.solve(8, 3)
    # test_agent.solve(250,100)
    # print(test_agent.solve(5, 5))
    print()


def get_runtimes(sheep, wolves):
    num_for_avg = 10
    number_of_sheep = sheep
    number_of_wolves = wolves
    run_time_df = pd.DataFrame(np.zeros(shape=(number_of_wolves, number_of_sheep)),
                               columns=[i for i in range(1, number_of_sheep + 1)],
                               index=[i for i in range(number_of_wolves, 0, -1)])
    test_agent = SemanticNetsAgent()
    for num_sheep in range(1, number_of_sheep + 1):
        for num_wolves in range(1, number_of_wolves + 1):
            times = np.zeros(shape=(num_for_avg,))
            for i in range(num_for_avg):
                start_time = time.time()
                test_agent.solve(num_sheep, num_wolves)
                end_time = time.time()
                elapsed_time = end_time - start_time
                times[i] = elapsed_time
            avg_time = np.mean(times)
            run_time_df.loc[num_wolves, num_sheep] = avg_time
    return run_time_df


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
    print("Heatmap Finished")
    return ax


if __name__ == "__main__":
    # print(sys.getrecursionlimit())
    # sys.setrecursionlimit(5000)
    test()
    fig, ax = plt.subplots(figsize=(12, 12))

    sheep, wolves = 40, 40
    all_run_time_df = get_runtimes(sheep, wolves)
    image = all_run_time_df
    heatmap(image,
            row_labels=np.arange(wolves, 0, -1),
            col_labels=np.arange(1, sheep + 1),
            ax=ax, cbarlabel="Seconds",
            title=f"Sheep Vs Wolves Runtimes",
            folder="",
            filename=f"Sheep_Wolves_Problem_Runtimes",
            cmap="inferno", set_label_tick_marks=True)
    ax.set_xlabel(f"Number of Sheep", fontsize=15, weight='heavy')
    ax.set_ylabel(f"Number of Wolves", fontsize=15, weight='heavy')
    plt.tight_layout()
    plt.savefig(f"Sheep_Wolves_Problem_Runtimes.png")
    print()

