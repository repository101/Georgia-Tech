# CSE-6242 Data and Visual Analytics
# Dr. Duen Horng(Polo) Chau
# Group Project - Baseball Pitch Prediction
# Josh Adams

# This was just written to combine the many .csv files we had

import pandas as pd
import glob


def combine_csv(input_files, year):
    try:
        print("Reading files in CSV/CSV_{}/ ".format(year))
        # Read and combine the many different .csv files in the specified directory
        data_frame = pd.concat([pd.read_csv(file, header=0) for file in input_files], ignore_index=True, sort=False)
        print("Writing {}_COMBINED_CSV.csv ".format(year))
        # Write the combined data to a single .csv file
        data_frame.to_csv("{}_COMBINED_CSV.csv".format(year), header=True)
        print("FINISHED")
        return
    except Exception as err:
        print("Something went wrong\n{}".format(err))


if __name__ == "__main__":

    # I specify a range which is used as part of the names in the CSV files it reads
    #   so that I am able to iterate over the different folders I had.
    for i in range(2018, 2019):
        directory = "CSV/CSV_{}/*.csv".format(str(i))

        # Get all the files in the directory that end in .csv
        files = glob.glob(directory)

        # Combine the specified .csv files
        combine_csv(files, str(i))

