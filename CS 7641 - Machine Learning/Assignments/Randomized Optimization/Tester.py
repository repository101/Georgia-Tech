import ro_ml_util as utl


if __name__ == '__main__':
    # gathered_data = utl.setup(["MNIST"])
    # gathered_data_fashion = utl.setup(["Fashion-MNIST"])
    # train_X, train_y, valid_X, valid_y, test_X, test_y = utl.split_data(gathered_data["MNIST"]["X"],
    #                                                                     gathered_data["MNIST"]["y"], minMax=True,
    #                                                                     oneHot=True)
    # fashion_train_X, fashion_train_y, fashion_valid_X, fashion_valid_y, fashion_test_X, fashion_test_y = utl.split_data(
    #     gathered_data_fashion["Fashion-MNIST"]["X"],
    #     gathered_data_fashion["Fashion-MNIST"]["y"], minMax=True)
    
    # utl.find_best_neural_network_sa(train_limit=200)
    # utl.find_best_neural_network_ga(train_limit=200)
    acc, time = utl.find_best_neural_network_rhc(train_limit=10000, num_iter=20)
    print()

