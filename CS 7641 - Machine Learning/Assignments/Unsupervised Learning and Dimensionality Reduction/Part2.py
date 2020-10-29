#!/usr/bin/env python
# coding: utf-8

# In[12]:

import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt

import unsupervised_learning_util as utl

plt.tight_layout()
plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500

NJOBS = 32
VERBOSE = 0
limit = 5000

folder = "DimensionalityReduction/"
utl.check_folder(folder)

# In[4]:


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

# # Dimensionality Reduction

# ## PCA

# In[5]:

# mnist_results = utl.run_pca(mnist["train_X"].iloc[:limit, :],
#                                                     keep_variance=0.99, dataset_name="MNIST")
# fashion_results = utl.run_pca(fashion_mnist["train_X"].iloc[:limit, :],
#                                                     keep_variance=0.99, dataset_name="Fashion")
# ica_results_mnist = utl.run_ica(mnist["train_X"].iloc[:limit, :],
#                                 20, load_pkl=False, dataset_name="MNIST")
# ica_results_fashion = utl.run_ica(fashion_mnist["train_X"].iloc[:limit, :],
#                                   20, load_pkl=False, dataset_name="Fashion_MNIST")
# random_projection_mnist = utl.run_randomized_projections(data=mnist["train_X"].iloc[:limit, :],
#                                                          dataset_name="MNIST", max_components=751, load_pkl=False)
# random_projection_fashion = utl.run_randomized_projections(data=fashion_mnist["train_X"].iloc[:limit, :],
#                                                            dataset_name="Fashion", max_components=751, load_pkl=False)
# random_forest_mnist = utl.run_random_forest(data_X=mnist["train_X"].iloc[:limit, :],
#                                             data_y=mnist["train_y"].iloc[:limit],
#                                             dataset_name="MNIST", max_components=751, load_pkl=False,
#                                             valid_X=mnist["valid_X"].iloc[:limit, :],
#                                             valid_y=mnist["valid_y"].iloc[:limit])
# random_forest_fashion = utl.run_random_forest(data_X=fashion_mnist["train_X"].iloc[:limit, :],
#                                               data_y=fashion_mnist["train_y"].iloc[:limit],
#                                               valid_X=fashion_mnist["valid_X"].iloc[:limit, :],
#                                               valid_y=fashion_mnist["valid_y"].iloc[:limit],
#                                               dataset_name="Fashion", max_components=751, load_pkl=False)

# with open(f"{os.getcwd()}/DimensionalityReduction/"
#           f"IndependentComponentAnalysis/MNIST_ICA_All_Pixel_Error.pkl", "rb") as input_file:
#     mnist_all_pixel_error = pickle.load(input_file)
#     input_file.close()
# with open(f"{os.getcwd()}/DimensionalityReduction/"
#           f"IndependentComponentAnalysis/MNIST_ICA_All_Obs_Error.pkl", "rb") as input_file:
#     mnist_all_obs_error = pickle.load(input_file)
#     input_file.close()

# with open(f"{os.getcwd()}/DimensionalityReduction/"
#           f"IndependentComponentAnalysis/Fashion_MNIST_ICA_All_Pixel_Error.pkl", "rb") as input_file:
#     fashion_all_pixel_error = pickle.load(input_file)
#     input_file.close()
# with open(f"{os.getcwd()}/DimensionalityReduction/"
#           f"IndependentComponentAnalysis/Fashion_MNIST_ICA_All_Obs_Error.pkl", "rb") as input_file:
#     fashion_all_obs_error = pickle.load(input_file)
#     input_file.close()

# with open(f"{os.getcwd()}/DimensionalityReduction/PrincipleComponentAnalysis/"
#           f"MNIST_PCA_Results.pkl", "rb") as input_file:
#     mnist_results = pickle.load(input_file)
#     input_file.close()
#
# with open(f"{os.getcwd()}/DimensionalityReduction/PrincipleComponentAnalysis/"
#           f"Fashion_PCA_Results.pkl", "rb") as input_file:
#     fashion_results = pickle.load(input_file)
#     input_file.close()

# with open(f"{os.getcwd()}/DimensionalityReduction/"
#           f"IndependentComponentAnalysis/MNIST_ICA_Results.pkl", "rb") as input_file:
#     ica_results_mnist = pickle.load(input_file)
#     input_file.close()
#
# with open(f"{os.getcwd()}/DimensionalityReduction/"
#           f"IndependentComponentAnalysis/Fashion_MNIST_ICA_Results.pkl", "rb") as input_file:
#     ica_results_fashion = pickle.load(input_file)
#     input_file.close()
#
# with open(f"{os.getcwd()}/DimensionalityReduction/"
#           f"RandomProjections/MNIST_RP_results.pkl", "rb") as input_file:
#     rp_results_mnist = pickle.load(input_file)
#     input_file.close()
#
# with open(f"{os.getcwd()}/DimensionalityReduction/"
#           f"RandomProjections/Fashion_RP_results.pkl", "rb") as input_file:
#     rp_results_fashion = pickle.load(input_file)
#     input_file.close()


# with open(f"{os.getcwd()}/DimensionalityReduction/RandomForest/RandomForest_MNIST_Results.pkl", "rb") as input_file:
#     rf_results_mnist = pickle.load(input_file)
#     input_file.close()
# with open(f"{os.getcwd()}/DimensionalityReduction/RandomForest/RandomForest_Fashion_Results.pkl", "rb") as input_file:
#     rf_results_fashion = pickle.load(input_file)
#     input_file.close()

# utl.plot_pca_results(mnist_results=mnist_results, fashion_results=fashion_results,
#                      mnist_X=mnist_not_scaled["train_X"].iloc[:limit, :],
#                      mnist_y=mnist_not_scaled["train_y"].iloc[:limit],
#                      fashion_X=fashion_mnist_not_scaled["train_X"].iloc[:limit, :],
#                      fashion_y=fashion_mnist_not_scaled["train_y"].iloc[:limit])

# utl.plot_ica_results(ica_results_mnist, ica_results_fashion,
#                      mnist_data=mnist["train_X"].iloc[:limit, :],
#                      fashion_data=fashion_mnist["train_X"].iloc[:limit, :],
#                      mnist_X=mnist_not_scaled["train_X"].iloc[:limit, :],
#                      mnist_y=mnist_not_scaled["train_y"].iloc[:limit],
#                      fashion_X=fashion_mnist_not_scaled["train_X"].iloc[:limit, :],
#                      fashion_y=fashion_mnist_not_scaled["train_y"].iloc[:limit],
#                      mnist_X_scaled=mnist["train_X"].iloc[:limit, :],
#                      mnist_y_scaled=mnist["train_y"].iloc[:limit],
#                      fashion_X_scaled=fashion_mnist["train_X"].iloc[:limit, :],
#                      fashion_y_scaled=fashion_mnist["train_y"].iloc[:limit])

# utl.plot_randomized_projection_results(rp_results_mnist, rp_results_fashion,
#                                        mnist_X=mnist_not_scaled["train_X"].iloc[:limit, :],
#                                        mnist_y=mnist_not_scaled["train_y"].iloc[:limit],
#                                        fashion_X=fashion_mnist_not_scaled["train_X"].iloc[:limit, :],
#                                        fashion_y=fashion_mnist_not_scaled["train_y"].iloc[:limit],
#                                        mnist_X_scaled=mnist["train_X"].iloc[:limit, :],
#                                        mnist_y_scaled=mnist["train_y"].iloc[:limit],
#                                        fashion_X_scaled=fashion_mnist["train_X"].iloc[:limit, :],
#                                        fashion_y_scaled=fashion_mnist["train_y"].iloc[:limit])

# utl.plot_random_forest_results(mnist_results=rf_results_mnist, fashion_results=rf_results_fashion,
#                                mnist_X=mnist_not_scaled["train_X"].iloc[:limit, :],
#                                mnist_y=mnist_not_scaled["train_y"].iloc[:limit],
#                                fashion_X=fashion_mnist_not_scaled["train_X"].iloc[:limit, :],
#                                fashion_y=fashion_mnist_not_scaled["train_y"].iloc[:limit])
# ## TSNE

# In[ ]:
