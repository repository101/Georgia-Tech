import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive
import seaborn as sns
import ro_ml_util as utl
import os
import pickle
import glob


if __name__ == '__main__':
    # utl.check_folder("NeuralNetwork/")
    # plt.style.use("ggplot")
    # folder = "NeuralNetwork/Gradient_Descent/"
    # utl.check_folder(folder)
    #
    # SEED = 42
    #
    # n_iterations = np.arange(1, 201, 1)
    # prob_size = "XL"
    # verbose = True
    # problem_name = "Gradient_Descent"
    # with open(f"{os.getcwd()}/{folder}/Final_Results_GD.pkl", "rb") as input_file:
    #     results_object = pickle.load(input_file)
    #     input_file.close()
    # fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))
    #
    # ax = utl.heatmap(results_object["lr_vs_iteration_train"],
    #                  row_labels=results_object["lr_vs_iteration_train"].index,
    #                  col_labels=results_object["lr_vs_iteration_train"].columns,
    #                  ax=ax1, cmap="viridis", cbarlabel="Training Accuracy",
    #                  x_label="Learning Rates",
    #                  y_label="Iterations",
    #                  title=f"Fashion-MNIST\nGradient Descent", folder=folder,
    #                  filename="Fashion_MNIST_Gradient_Descent_LR_VS_Iteration")
    # print()

    # gathered_data = utl.setup(["MNIST"])
    # gathered_data_fashion = utl.setup(["Fashion-MNIST"])
    # train_X, train_y, valid_X, valid_y, test_X, test_y = utl.split_data(gathered_data["MNIST"]["X"],
    #                                                                     gathered_data["MNIST"]["y"], minMax=True,
    #                                                                     oneHot=True)
    # fashion_train_X, fashion_train_y, fashion_valid_X, fashion_valid_y, fashion_test_X, fashion_test_y = utl.split_data(
    #     gathered_data_fashion["Fashion-MNIST"]["X"],
    #     gathered_data_fashion["Fashion-MNIST"]["y"], minMax=True)
    iteration_list = (10 ** np.arange(1, 2, 1)).tolist()
    # rhc_params = {"restart_list": (2 ** np.arange(1, 4, 1)).tolist(),
    #               "iterations": (2 ** np.arange(1, 9, 1)).tolist(),
    #               "learning_rate": (10. ** np.arange(-3, 4, 1)).tolist(),
    #               "max_attempts": (2 ** np.arange(1, 4, 1)).tolist()}
    
    rhc_params = {"iterations": [1000, 2000, 3000, 4000, 5000],
                  "learning_rates": [0.01, 0.1, 1.0, 5.0]}
    
    sa_params = {"iterations": [1000, 2000, 3000, 4000, 5000],
                 "learning_rates": [0.01, 0.1, 1.0, 5.0]}
    
    ga_params = {"iterations": [1000, 2000, 3000, 4000, 5000],
                 "learning_rate": [0.01, 0.1, 1.0, 5.0]}
    
    mimic_params = {"population_sizes": np.arange(50, 101, 50).tolist(),
                    "keep_percent_list": np.round(np.arange(0.30, 0.41, 0.1), 2).tolist(),
                    "iterations": (2 ** np.arange(1, 2, 1)).tolist()}
    
    # best_nn_results = utl.find_best_neural_network_gradient_descent(
    #     train_limit=1000,
    #     verbose=True,
    #     iterations=(2. ** np.arange(1, 5, 1)).tolist(),
    #     attempts=np.arange(1, 5, 1).astype(np.int).tolist(),
    #     learning_rates=np.round(10. ** np.arange(-5, 4, 1), 6))
    
    # d = utl.find_best_neural_network_rhc(train_limit=1000, verbose=True, rhc_parameters=rhc_params)
    b = utl.find_best_neural_network_sa(train_limit=1000, verbose=True, sa_parameters=sa_params)
    c = utl.find_best_neural_network_ga(train_limit=1000, verbose=True, ga_parameters=ga_params)

    print()

