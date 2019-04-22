# CSE-6242 Data and Visual Analytics
# Dr. Duen Horng(Polo) Chau
# Group Project - Baseball Pitch Prediction using Recurrent Neural Networks
# Josh Adams

import pandas as pd
import glob


def Remove_Nan(input_file, keep_pitcher_column=False):
    # Broke this out because other script was getting messy

    # Reads a file as a csv
    the_dataframe = pd.read_csv(input_file, sep=',', header=0)
    # Splits the input file at \\ because that was a way to get my file name
    split_file_name = input_file.split("\\")
    # Establishing a new file name
    output_filename = "Data_New/{}".format(split_file_name[1])
    initial_length = len(the_dataframe["pitch_type"])
    # Remove any row with NA
    the_dataframe.dropna(inplace=True)
    # Resetting the index or there will essentially be spaces where the removed rows were
    the_dataframe.reset_index(drop=True, inplace=True)

    # If keep_pitch_column is true, then we keep it in the dataset
    if keep_pitcher_column:
        the_dataframe.to_csv(output_filename, sep=",", index=False)
    else:
        try:
            if "pitcher" in the_dataframe.columns:
                # Saving the new dataframe with the pitcher column remove
                the_dataframe.drop("pitcher", axis=1).to_csv(output_filename, sep=",", index=False)
            else:
                # If the dataset does not contain the pitcher column such as if we loaded a file that did not
                #    contain that column
                the_dataframe.to_csv(output_filename, sep=",", index=False)
        except Exception as err:
            print("Failed when removing pitcher column from dataframe: ", err)
    print("Data for pitcher data is saved to {}".format(output_filename))

    return


if __name__ == "__main__":

    # This is the directory where I had the different .csv files
    directory = "Data/*.csv"
    files = glob.glob(directory)

    # Iterate over all of the files in that directory to remove the nan rows
    for file_name in files:
        Remove_Nan(file_name, keep_pitcher_column=False)
    print("\n\nFinished\n")
