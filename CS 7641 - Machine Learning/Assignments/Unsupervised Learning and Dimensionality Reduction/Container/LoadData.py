import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler


def LoadData(path_to_csv, normalize=False, size=None):
    # Verify path_to_csv ends in .csv

    try:
        print("Attempting to load: {}\n".format(path_to_csv.split("/")[-1]))
        if size is None:
            raw_combined_data = np.loadtxt(path_to_csv, delimiter=",", dtype=np.uint8)
        else:
            raw_combined_data = np.loadtxt(path_to_csv, delimiter=",", dtype=np.uint8, max_rows=size)
        print("Loading Complete")
        print("Data Statistics: \n"
              "   Number of Entries: {} \n"
              "   Shape of Entry: {}\n".format(raw_combined_data.shape[0], raw_combined_data[0].shape))
        labels = raw_combined_data[:, 0]
        data = raw_combined_data[:, 1::]

        if normalize:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)

            raw_combined_data = raw_combined_data.astype(np.float64)
            raw_combined_data[:, 1::] = data
        
        return labels, data, raw_combined_data
    
    except Exception as LoadDataException:
        print("Exception occurred while attempting to load the data: \n", LoadDataException)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
