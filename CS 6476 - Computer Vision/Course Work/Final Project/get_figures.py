import pickle
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from data_util import evenly_distribute_classes, df_to_array

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500
plt.tight_layout()
np.random.seed(42)


def get_data_from_pickle(filename, is_testing=False):
    with open(filename, "rb") as input_file:
        data = pickle.load(input_file)
    input_file.close()
    new_data = {}
    if is_testing:
        new_data = np.asarray(data)
        new_data = new_data[:, 1]
    else:
        for i in range(len(data)):
            for key, val in data[i].items():
                temp_key_min = key + "_min"
                temp_key_max = key + "_max"
                temp_key_avg = key + "_avg"
                if temp_key_min not in new_data:
                    new_data[temp_key_min] = []
                    new_data[temp_key_min].append(val[0])
                else:
                    new_data[temp_key_min].append(val[0])
                    
                if temp_key_max not in new_data:
                    new_data[temp_key_max] = []
                    new_data[temp_key_max].append(val[-1])
                else:
                    new_data[temp_key_max].append(val[-1])   
                    
                if temp_key_avg not in new_data:
                    new_data[temp_key_avg] = []
                    new_data[temp_key_avg].append(np.mean(val))
                else:
                    new_data[temp_key_avg].append(np.mean(val))  

    return new_data


def plot_single_learning_curve(dataframe, title="", x_label="", y_label="", save_filename=None, ax=None,
                               what_to_plot=["loss", "val_loss", "accuracy", "test", "val_accuracy"], show_title=True,
                               show_legend=True, base_for_legend="", show_grid=True, 
                               set_y_lim=True, plot_secondary=False,
                               secondary_plot=["loss", "val_loss", "accuracy", "test", "val_accuracy"],
                               ax2=None, ylim=[0, 1.01]):
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
    all_lns = []
    if not plot_secondary:
        if ax is None:
            plt.close("all")
            plt.style.use("ggplot")
            _, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        if set_y_lim:
            ax.set_ylim(ylim[0], ylim[1])
        if x_label != "":
            ax.set_xlabel(x_label)
            
        if y_label != "":
            ax.set_ylabel(y_label)
            
        if "loss" in what_to_plot:
            ax.fill_between(np.arange(dataframe["loss_avg"].shape[0]),
                             dataframe["loss_avg"] - np.abs(dataframe["loss_avg"] - dataframe["loss_min"]),
                             dataframe["loss_avg"] + np.abs(dataframe["loss_avg"] - dataframe["loss_max"]), alpha=0.2,
                             color="darkorange")
            lns = ax.plot(dataframe["loss_avg"], 'o-', color="darkorange", label=base_for_legend + "Training Loss")
            all_lns.append(lns)
            
        if "val_loss" in what_to_plot:
            ax.fill_between(np.arange(dataframe["val_loss_avg"].shape[0]),
                            dataframe["val_loss_avg"] - np.abs(dataframe["val_loss_avg"] - dataframe["val_loss_min"]),
                            dataframe["val_loss_avg"] + np.abs(dataframe["val_loss_avg"] - dataframe["val_loss_max"]),
                            alpha=0.2, color="darkorange")
            lns = ax.plot(dataframe["val_loss_avg"], 'o-', color="darkorange", label=base_for_legend + "Cross-validation Loss")
            all_lns.append(lns)
            
        if "val_accuracy" in what_to_plot:
            ax.fill_between(np.arange(dataframe["val_accuracy_avg"].shape[0]),
                             dataframe["val_accuracy_avg"] - np.abs(
                                 dataframe["val_accuracy_avg"] - dataframe["val_accuracy_min"]),
                             dataframe["val_accuracy_avg"] + np.abs(
                                 dataframe["val_accuracy_avg"] - dataframe["val_accuracy_max"]),
                             alpha=0.2, color="navy")
            lns = ax.plot(dataframe["val_accuracy_avg"], 'o-', color="darkgreen",
                     label=base_for_legend + "Cross-validation Accuracy")
            all_lns.append(lns)

        if "accuracy" in what_to_plot:
            ax.fill_between(np.arange(dataframe["accuracy_avg"].shape[0]),
                             dataframe["accuracy_avg"] - np.abs(dataframe["accuracy_avg"] - dataframe["accuracy_min"]),
                             dataframe["accuracy_avg"] + np.abs(dataframe["accuracy_avg"] - dataframe["accuracy_max"]),
                             alpha=0.2, color="navy")
            lns = ax.plot(dataframe["accuracy_avg"], 'o-', color="navy", label=base_for_legend + "Training Accuracy")
            all_lns.append(lns)

        if "test" in what_to_plot:
            lns = ax.plot(dataframe["testing"], 'o-', color="darkred", label=base_for_legend + "Testing Accuracy")
            all_lns.append(lns)
            
    else:
        if set_y_lim:
            ax2.set_ylim(ylim[0], ylim[1])
        if y_label != "":
            ax2.set_ylabel(y_label)
            
        if "loss" in secondary_plot:
            ax2.fill_between(np.arange(dataframe["loss_avg"].shape[0]),
                            dataframe["loss_avg"] - np.abs(dataframe["loss_avg"] - dataframe["loss_min"]),
                            dataframe["loss_avg"] + np.abs(dataframe["loss_avg"] - dataframe["loss_max"]), alpha=0.2,
                            color="darkorange")
            lns = ax2.plot(dataframe["loss_avg"], 'o-', color="darkorange", label=base_for_legend + "Training Loss")
            all_lns.append(lns)

        if "val_loss" in secondary_plot:
            ax2.fill_between(np.arange(dataframe["val_loss_avg"].shape[0]),
                            dataframe["val_loss_avg"] - np.abs(dataframe["val_loss_avg"] - dataframe["val_loss_min"]),
                            dataframe["val_loss_avg"] + np.abs(dataframe["val_loss_avg"] - dataframe["val_loss_max"]),
                            alpha=0.2, color="darkorange")
            lns = ax2.plot(dataframe["val_loss_avg"], 'o-', color="darkorange",
                    label=base_for_legend + "Cross-validation Loss")
            all_lns.append(lns)
        
        if "val_accuracy" in secondary_plot:
            ax2.fill_between(np.arange(dataframe["val_accuracy_avg"].shape[0]),
                            dataframe["val_accuracy_avg"] - np.abs(
                                dataframe["val_accuracy_avg"] - dataframe["val_accuracy_min"]),
                            dataframe["val_accuracy_avg"] + np.abs(
                                dataframe["val_accuracy_avg"] - dataframe["val_accuracy_max"]),
                            alpha=0.2, color="navy")
            lns = ax2.plot(dataframe["val_accuracy_avg"], 'o-', color="darkgreen",
                    label=base_for_legend + "Cross-validation Accuracy")
            all_lns.append(lns)
            
        if "accuracy" in secondary_plot:
            ax2.fill_between(np.arange(dataframe["accuracy_avg"].shape[0]),
                            dataframe["accuracy_avg"] - np.abs(dataframe["accuracy_avg"] - dataframe["accuracy_min"]),
                            dataframe["accuracy_avg"] + np.abs(dataframe["accuracy_avg"] - dataframe["accuracy_max"]),
                            alpha=0.2, color="navy")
            lns = ax2.plot(dataframe["accuracy_avg"], 'o-', color="navy", label=base_for_legend + "Training Accuracy")
            all_lns.append(lns)
            
        if "test" in secondary_plot:
            lns = ax2.plot(dataframe["testing"], 'o-', color="darkred", label=base_for_legend + "Testing Accuracy")
            all_lns.append(lns)

    if show_legend:
        ax.legend(loc="best", markerscale=1.1, frameon=True,
                  edgecolor="black", fancybox=True, shadow=True)
    # if show_grid:
    #     ax.grid(which='major', linestyle='-', linewidth='0.5', color='white')
    plt.tight_layout()
        
    # if save_filename is not None:
    #     plt.savefig(save_filename)
    return all_lns
    
    
def graph_stuff(unedited_training_results_dataframe, best_model, temp_test_x, temp_test_y, network_name, reshape_size):
    plt.close("all")
    fig = plt.figure(figsize=(20, 10))
    grid = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.2)
    ax1 = fig.add_subplot(grid[0, 0:2])
    # ax1 = plt.Subplot(fig, grid[0, 0:2])
    ax2 = ax1.twinx()
    ax1.grid(False)
    ax2.grid(False)
    # lns2 = plot_single_learning_curve(unedited_training_results_dataframe, title=None,
    #                            ax=ax1, show_title=False,
    #                            show_legend=False, y_label="Loss",
    #                            secondary_plot=["loss"],
    #                            show_grid=False, plot_secondary=True, ax2=ax2, ylim=[0.0, 1.01])
    lns2 = plot_single_learning_curve(unedited_training_results_dataframe, title=None, x_label="Session",
                                      ax=ax1, show_title=False, show_legend=False, y_label="Loss",
                                      what_to_plot=["loss"], show_grid=False, plot_secondary=False,
                                      ylim=[0.0, 1.01])
    
    # lns1 = plot_single_learning_curve(unedited_training_results_dataframe, title=None, x_label="Session",
    #                            y_label="Accuracy", ax=ax1, show_title=False,
    #                            show_legend=False,
    #                            what_to_plot=["accuracy", "test", "val_accuracy"],
    #                            show_grid=True, plot_secondary=False, ylim=[0.6, 1.01])
    lns1 = plot_single_learning_curve(unedited_training_results_dataframe, title=None, y_label="Accuracy", ax=ax1,
                                      show_title=False, show_legend=False,
                                      secondary_plot=["accuracy", "test", "val_accuracy"],
                                      show_grid=False, plot_secondary=True, ylim=[0.6, 1.01], ax2=ax2)
       
    all_lns = lns1 + lns2
    all_lns = np.ravel(np.asarray(all_lns)).tolist()
    fig.legend(loc="upper left", markerscale=1.1, frameon=True, edgecolor="black", fancybox=True, shadow=True)
    # ax2.legend(all_lns, [l.get_label() for l in all_lns], loc=0, markerscale=1.1, frameon=True,
    #            edgecolor="black", fancybox=True, shadow=True)
    ax3 = plt.Subplot(fig, grid[0, 2])
    ax1.set_title(f"{network_name} Learning Curve", fontsize=25, weight='bold', y=1.08)
    ax3.set_title("Negative Examples", fontsize=25, weight='bold', y=1.08)
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='white')
    ax3.axis('off')
    fig.add_subplot(ax3)
    
    # grid_left = grid[0, 0:1]
    grid_right = grid[0, 2].subgridspec(3, 3, hspace=0.3, wspace=0.1)
    y_pred = best_model.predict(temp_test_x)
    y_pred_converted = np.argmax(y_pred, axis=1)
    idx_of_incorrect = np.argwhere(y_pred_converted != temp_test_y)[:, 0]
    incorrect_images = temp_test_x[idx_of_incorrect]
    incorrect_pred_labels = y_pred_converted[idx_of_incorrect]
    actual_labels_for_incorrect_pred = temp_test_y[idx_of_incorrect]
    missed_labels = np.unique(incorrect_pred_labels, return_counts=True)
    missed_labels_percent = missed_labels[1] / np.sum(missed_labels[1])
    titles = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
              6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
    count = 1
    for i in range(3):
        for j in range(3):
            temp_ax = fig.add_subplot(grid_right[i, j], autoscale_on=True, frameon=True)
            temp_idx = actual_labels_for_incorrect_pred == count
            temp_ax.set_xticks([])
            temp_ax.set_yticks([])
            incorrect_img_to_show = incorrect_images[temp_idx][0, :]
            t_pred_label = incorrect_pred_labels[temp_idx][0]
            t_im = np.reshape(incorrect_img_to_show, newshape=(reshape_size[1], reshape_size[0], 3)).astype(np.uint8)
            plt.imshow(t_im)
            plt.xlabel(f"Predicted: {t_pred_label}", fontsize=15)
            plt.title(f"{titles[count]}", fontsize=15)
            fig.add_subplot(temp_ax)
            count += 1
    
    plt.savefig(f"{network_name}_LearningCurve_With_Negative_Examples.png", bbox_inches='tight')
    plt.close("all")
    return


def get_lenet_graph():
    network_name = "LeNet5"
    reshape_size = (32, 64, 3)
    testing_data = pd.read_hdf("test_data.hd5", key="test")
    test_x, test_y = df_to_array(testing_data, img_shape=(reshape_size))
    temp_test_x, temp_test_y = evenly_distribute_classes(test_x, test_y)
    unedited_training_results = get_data_from_pickle(f"Trained_Model_{network_name}_Final_Sess_50.pkl")
    unedited_testing_results = get_data_from_pickle(f"Trained_Model_LeNet5_Acc_0.921_Final_Sess_50_Testing_Results.pkl", is_testing=True)
    best_model = tf.keras.models.load_model("Trained_Model_LeNet5_Acc_0.922_Sess_44.h5")
    for key, val in unedited_training_results.items():
        unedited_training_results[key] = np.asarray(val)
    unedited_training_results_dataframe = pd.DataFrame(data=unedited_training_results)
    unedited_training_results_dataframe["testing"] = unedited_testing_results
    graph_stuff(unedited_training_results_dataframe=unedited_training_results_dataframe, best_model=best_model,
                temp_test_x=temp_test_x, temp_test_y=temp_test_y, network_name=network_name, reshape_size=reshape_size)
    print(f"Finished {network_name}")
    return


def get_resnet_graph():
    network_name = "ResNet"
    reshape_size = (32, 64, 3)
    testing_data = pd.read_hdf("test_data.hd5", key="test")
    test_x, test_y = df_to_array(testing_data, img_shape=(reshape_size))
    temp_test_x, temp_test_y = evenly_distribute_classes(test_x, test_y)
    unedited_training_results = get_data_from_pickle(f"Trained_Model_ResNet_Final_Sess_50.pkl")
    unedited_training_results_dataframe = pd.DataFrame(unedited_training_results)
    unedited_testing_results = get_data_from_pickle("Trained_Model_ResNet_Acc_0.928_Final_Sess_50_Testing_Results.pkl", is_testing=True)
    best_model = tf.keras.models.load_model(f"Trained_Model_ResNet_Acc_0.928_Sess_50.h5")
    for key, val in unedited_training_results.items():
        unedited_training_results[key] = np.asarray(val)
        
    unedited_training_results_dataframe = pd.DataFrame(data=unedited_training_results)
    unedited_training_results_dataframe["testing"] = unedited_testing_results
    graph_stuff(unedited_training_results_dataframe=unedited_training_results_dataframe, best_model=best_model,
                temp_test_x=temp_test_x, temp_test_y=temp_test_y, network_name=network_name, reshape_size=reshape_size)
    print(f"Finished {network_name}")
    return


def get_vgg16_graph():
    network_name = "VGG16"
    reshape_size = (32, 64, 3)
    testing_data = pd.read_hdf("test_data.hd5", key="test")
    test_x, test_y = df_to_array(testing_data, img_shape=(reshape_size))
    temp_test_x, temp_test_y = evenly_distribute_classes(test_x, test_y)
    unedited_training_results = get_data_from_pickle(f"Trained_Model_VGG16_Acc_0.957_Sess_49.pkl")
    unedited_testing_results = get_data_from_pickle(f"Trained_Model_VGG16_Acc_0.957_Final_Sess_50_Testing_Results.pkl", is_testing=True)
    best_model = tf.keras.models.load_model(f"Trained_Model_VGG16_Acc_0.957_Sess_49.h5")
    for key, val in unedited_training_results.items():
        unedited_training_results[key] = np.asarray(val)
    unedited_training_results_dataframe = pd.DataFrame(data=unedited_training_results)
    unedited_training_results_dataframe["testing"] = unedited_testing_results
    graph_stuff(unedited_training_results_dataframe=unedited_training_results_dataframe, best_model=best_model,
                temp_test_x=temp_test_x, temp_test_y=temp_test_y, network_name=network_name, reshape_size=reshape_size)
    print(f"Finished {network_name}")
    return


if __name__ == "__main__":
    get_lenet_graph()
    get_vgg16_graph()
    get_resnet_graph()

    exit()
