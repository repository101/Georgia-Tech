import os
import sys
import time

import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Your dataset which we are finding 10 clusters
    my_dataset

    results = kmean(cluster=10).fit_predict(my_dataset)

    # Get Sum of Squared error on whole set
    get_sse(results)

    # Get Sum of Square error for just the cluster of label 0
    get_sse(resuls.iloc[:, "cluster_label" == 0])

    print()
