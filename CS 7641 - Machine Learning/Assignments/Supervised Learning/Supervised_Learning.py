import numpy as np

from Machine_Learning import SupervisedLearning
from ml_util import setup

if __name__ == "__main__":
    np.random.seed(42)
    # gathered_data = setup(["MNIST", "Fashion-MNIST"])
    gathered_data = setup(["MNIST"])
    print()

    parameters = {"NN": {"epochs": 2},
                  "DT": {"max_depth": 20,
                         "min_samples_split": 2,
                         "min_samples_leaf": 1},
                  "Boost": {"n_estimators": 50,
                            "learning_rate": 1,
                            "algorithm": "SAMME"},
                  "SVM": {"C": 1.0,
                          "kernel": "rbf",
                          "degree": 3,
                          "decision_function_shape": "ovo"},
                  "KNN": {"n_neighbors": 5,
                          "algorithm": "kd_tree",
                          "leaf_size": 5}}

    SupervisedLearning(data=gathered_data["MNIST"]["Full_Dataframe"],
                       x=gathered_data["MNIST"]["X"],
                       y=gathered_data["MNIST"]["y"],
                       parameters=parameters)

    print()
