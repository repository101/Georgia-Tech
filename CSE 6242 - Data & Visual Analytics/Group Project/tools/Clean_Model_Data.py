# CSE-6242 Data and Visual Analytics
# Dr. Duen Horng(Polo) Chau
# Group Project - Baseball Pitch Prediction
# Josh Adams

import pandas as pd
import numpy as np
import pickle
import math

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix



def Improve_Network():
    # This will be time consuming....
    # Used to iterate through hundreds of combinations of parameters for the neural networks and find the
    #    best performing set

    model_name = "112526"
    test_file = "Data/{}.csv".format(model_name)

    layer_one = 20
    layer_two = 20
    layer_three = 20
    layer_four = 1
    layer_five = 1
    layer_1_value = 1
    layer_2_value = 1
    layer_3_value = 1
    layer_4_value = 0
    max_iterations = 100

    # VERY TIME CONSUMING
    # Do Not Use
    for layer_5_value in range(layer_five):
        print("Layer Values: {} {} {} {} {} \n".format(layer_1_value, layer_2_value, layer_3_value,
                                                       layer_4_value, layer_5_value))
        test_network, weighted_score = Generate_Neural_Network(filename=test_file, split_percentage=0.3,
                                                               data_scaler=data_scaler,
                                                               layer_one_size=layer_1_value,
                                                               layer_two_size=layer_2_value,
                                                               layer_three_size=layer_3_value,
                                                               layer_four_size=layer_4_value,
                                                               layer_five_size=layer_5_value,
                                                               max_iterations=max_iterations)

        Save_Network(test_network, model_name, max_iterations, weighted_score, layer_1_value, layer_2_value,
                     layer_3_value, layer_4_value, layer_5_value)
        for layer_4_value in range(layer_four):
            print("Layer Values: {} {} {} {} \n".format(layer_1_value, layer_2_value, layer_3_value,
                                                        layer_4_value))
            test_network, weighted_score = Generate_Neural_Network(filename=test_file, split_percentage=0.3,
                                                                   data_scaler=data_scaler,
                                                                   layer_one_size=layer_1_value,
                                                                   layer_two_size=layer_2_value,
                                                                   layer_three_size=layer_3_value,
                                                                   layer_four_size=layer_4_value,
                                                                   layer_five_size=0,
                                                                   max_iterations=max_iterations)

            Save_Network(test_network, model_name, max_iterations, weighted_score, layer_1_value, layer_2_value,
                         layer_3_value,
                         layer_4_value, layer_5_value)
            for layer_3_value in range(1, layer_three, 3):
                print("Layer Values: {} {} {} \n".format(layer_1_value, layer_2_value, layer_3_value))
                test_network, weighted_score = Generate_Neural_Network(filename=test_file, split_percentage=0.3,
                                                                       data_scaler=data_scaler,
                                                                       layer_one_size=layer_1_value,
                                                                       layer_two_size=layer_2_value,
                                                                       layer_three_size=layer_3_value,
                                                                       layer_four_size=0,
                                                                       layer_five_size=0,
                                                                       max_iterations=max_iterations)

                Save_Network(test_network, model_name, max_iterations, weighted_score, layer_1_value, layer_2_value, layer_3_value,
                             layer_4_value, layer_5_value)
                for layer_2_value in range(1, layer_two, 3):
                    print("Layer Values: {} {} {} \n".format(layer_1_value, layer_2_value, layer_3_value))
                    test_network, weighted_score = Generate_Neural_Network(filename=test_file, split_percentage=0.3,
                                                                           data_scaler=data_scaler,
                                                                           layer_one_size=layer_1_value,
                                                                           layer_two_size=layer_2_value,
                                                                           layer_three_size=0,
                                                                           layer_four_size=0,
                                                                           layer_five_size=0,
                                                                           max_iterations=max_iterations)

                    Save_Network(test_network, model_name, max_iterations, weighted_score, layer_1_value, layer_2_value,
                                 layer_3_value,
                                 layer_4_value, layer_5_value)
                    for layer_1_value in range(14, layer_one, 3):
                        print("Layer Values: {} {} {} \n".format(layer_1_value, layer_2_value, layer_3_value))
                        test_network, weighted_score = Generate_Neural_Network(filename=test_file, split_percentage=0.3,
                                                                               data_scaler=data_scaler,
                                                                               layer_one_size=layer_1_value,
                                                                               layer_two_size=0, layer_three_size=0,
                                                                               layer_four_size=0, layer_five_size=0,
                                                                               max_iterations=max_iterations)

                        Save_Network(test_network, model_name, max_iterations, weighted_score, layer_1_value,
                                     layer_2_value, layer_3_value,
                                     layer_4_value, layer_5_value)



                        # test_network, weighted_score = Generate_Neural_Network(filename=test_file, split_percentage=0.3,
    #                                                        data_scaler=data_scaler, layer_one_size=layer_one,
    #                                                        layer_two_size=layer_two, layer_three_size=layer_three,
    #                                                        layer_four_size=layer_four, layer_five_size=layer_five,
    #                                                        max_iterations=max_iterations)
    #
    # Save_Network(test_network, model_name, layer_one, layer_two, layer_three, layer_four, layer_five, max_iterations,
    #              weighted_score)
    return


def Save_Network(the_network_to_save, network_name, max_iter, score, l1=0, l2=0, l3=0, l4=0, l5=0):
    # I want all to be saved in "Trained_Networks/"
    directory = "Trained_Networks"
    if (l4 == 0) and (l5 == 0):
        pickle.dump(the_network_to_save, open("{}/{}_{}_Network_{}-{}-{}--{}.sav".format(directory, network_name,
                                                                                         score, l1, l2,
                                                                                         l3, max_iter), "wb"))
    elif l5 == 0:
        pickle.dump(the_network_to_save, open("{}/{}_{}_Network_{}-{}-{}-{}--{}.sav".format(directory, network_name,
                                                                                            score, l1, l2,
                                                                                            l3, l4, max_iter), "wb"))
    else:
        pickle.dump(the_network_to_save, open("{}/{}_{}_Network_{}-{}-{}-{}-{}--{}.sav".format(directory, network_name,
                                                                                               score, l1, l2,
                                                                                               l3, l4, l5,
                                                                                               max_iter), "wb"))
    return


def Determine_Pitch_Type_To_Keep_Pitcher_Specific(the_pitcher_id, the_data=None):
    print("Determining the PitchTypes to use with pitcher {}\n".format(the_pitcher_id))
    # Getting count for each type of pitch
    pitch_type_count_dict = the_data.pitch_type.value_counts()
    num_of_pitch = len(the_data["pitch_type"])
    list_of_pitch_types_used = list(the_data.pitch_type.unique())
    threshold = 0.02
    knuckle_thresh = 0.5

    # Remove woba and swstrike columns relating to pitch types not thrown by the pitcher
    for pitchType in all_pitch_types:
        if pitchType not in list_of_pitch_types_used:
            woba_column_to_remove = "woba.{}".format(pitchType)
            swstrike_column_to_remove = "swstrike_pct.{}".format(pitchType)
            if woba_column_to_remove in the_data.columns:
                the_data.drop(woba_column_to_remove, axis=1, inplace=True)
            if swstrike_column_to_remove in the_data.columns:
                the_data.drop(swstrike_column_to_remove, axis=1, inplace=True)

    for key, value in pitch_type_count_dict.iteritems():
        current_pitch_type_percentage_of_total = value/(len(the_data["pitch_type"]))
        # If the pitcher throws more than knuckle_thresh, KN, then we remove all woba and swstrike columns
        if (key == "KN") and (current_pitch_type_percentage_of_total > knuckle_thresh):
            current_pitch_types = list(the_data.pitch_type.unique())
            for pitch in current_pitch_types:
                if pitch in the_data.columns:
                    the_data.drop(pitch, axis=1, inplace=True)
                if pitch in the_data.columns:
                    the_data.drop(pitch, axis=1, inplace=True)
        # Finds and removes pitch types if they have not been used enough by the pitcher
        #   specified by the threshold
        if current_pitch_type_percentage_of_total < threshold:
            print("Pitch total {}, current pitch type {} and total {}, percentage {}".format(
                (len(the_data["pitch_type"])), key, value, current_pitch_type_percentage_of_total))
            the_data = the_data[the_data.pitch_type != key]
            print("The number of pitches now {}".format(len(the_data["pitch_type"])))

    return


def Get_Pitcher_Data(pitcher_id, dataframe, save_directory, keep_pitcher_column=False, test_run_pitcher_data=False):
    # Separated the dataframe by pitcher id
    pitcher_id = int(pitcher_id)
    temp = dataframe.loc[dataframe["pitcher"] == pitcher_id]
    if test_run_pitcher_data:
        # This was used to test different formats
        file_name = str(save_directory) + str(pitcher_id) + ".csv"
        Determine_Pitch_Type_To_Keep_Pitcher_Specific(pitcher_id, temp)
        temp.to_csv(file_name, sep=',', index=False)

    else:
        file_name = str(save_directory) + str(pitcher_id) + ".csv"
        Determine_Pitch_Type_To_Keep_Pitcher_Specific(pitcher_id, temp)
        # We should save all data except the pitcher id because it is in the file name,
        #    thus all data should be assumed to belong to that id.
        if keep_pitcher_column:
            temp.to_csv(file_name, sep=",", index=False)
        else:
            temp.drop("pitcher", axis=1).to_csv(file_name, sep=",", index=False)
        print("Data for pitcher id {} is saved to {}".format(str(pitcher_id), file_name))

    return


def Replace_Value_in_Column(old_value, new_value, column_name, dataframe):
    # This will replace a passed in value in a column, with a new value
    print("\nThe value {} will be replaced with {} in column {}\n".format(old_value, new_value, column_name))
    return dataframe[column_name].replace(old_value, new_value, inplace=True)


def Separate_Pitchers(dataframe, save_directory, keep_pitcher_column=False, test_run_sep_pitch=False):
    # This will separate the data by pitchers
    if test_run:
        print("This is a test run.\nIf this is not a test run, change 'test_run' parameter.")
    print("\nSeparating the data by pitcher id \n")
    pitcher_ids = dataframe.pitcher.unique()
    # This was used to get a list of all of the pitchers in the dataset
    # the_file = open("Pitcher_Ids.txt", "w")
    # for k in pitcher_ids:
    #     the_file.write(str(k) + "\n")
    # the_file.close()
    count = 0
    number_of_ids = len(pitcher_ids)
    # Iterate over the pitchers ids and process them by passing the information to Get_Pitcher_Data
    for id_of_pitcher in pitcher_ids:
        count += 1
        if count % int((number_of_ids/100)) == 0:
            print("{}% Completed \n".format(count/int(math.ceil((number_of_ids/100)))))
        Get_Pitcher_Data(id_of_pitcher, dataframe, save_directory, keep_pitcher_column, test_run_sep_pitch)
    print("\nFinished Separating Pitcher Data \n\n")
    return


def Encode_Categorical_Data(dataframe, column_name, encoder):
    # Passed in a dataframe and column, then encodes the column to numerical values
    print("\nEncoding {} ".format(column_name))
    dataframe[column_name] = encoder.fit_transform(dataframe[column_name])
    return


def Generate_Neural_Network(filename, split_percentage=0.30, data_scaler=StandardScaler(),
                            layer_one_size=5, layer_two_size=5, layer_three_size=5, layer_four_size=0,
                            layer_five_size=0, max_iterations=50, alpha=1e-5, display_results=False
                            ):
    current_dataframe = pd.read_csv(filename, sep=",", header=0, low_memory=False)
    print("Data has been loaded for {}".format(filename))

    # Initialize the X and y variables
    print("X has been initialized: \n")
    X = current_dataframe.drop("pitch_type", axis=1)
    print("y has been initialized: \n")
    y = current_dataframe["pitch_type"]

    # Split the data
    if split_percentage > 1:
        split_percentage = 0.3
    print("Data is being split {}/{} test/train \n".format((1 - split_percentage), split_percentage))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_percentage)

    # Scale the data
    print("Scaling the data \n")
    data_scaler.fit(X_train)

    # Initialize the neural network
    print("Initializing the Neural Network \n")
    if (layer_two_size > 0) and (layer_three_size > 0) and (layer_four_size > 0) and (layer_five_size > 0):
        the_network = MLPClassifier(hidden_layer_sizes=(layer_one_size, layer_two_size, layer_three_size,
                                                        layer_four_size, layer_five_size), max_iter=max_iterations,
                                    alpha=alpha)
    elif (layer_two_size > 0) and (layer_three_size > 0) and (layer_four_size > 0):
        the_network = MLPClassifier(hidden_layer_sizes=(layer_one_size, layer_two_size, layer_three_size,
                                                        layer_four_size), max_iter=max_iterations,
                                    alpha=alpha)
    elif (layer_two_size > 0) and (layer_three_size > 0):
        the_network = MLPClassifier(hidden_layer_sizes=(layer_one_size, layer_two_size, layer_three_size),
                                    max_iter=max_iterations,
                                    alpha=alpha)
    elif (layer_two_size > 0):
        the_network = MLPClassifier(hidden_layer_sizes=(layer_one_size, layer_two_size),
                                    max_iter=max_iterations,
                                    alpha=alpha)
    else:
        the_network = MLPClassifier(hidden_layer_sizes=(layer_one_size), max_iter=max_iterations, alpha=alpha)

    # Train the network
    print("Training the classifier \n")
    the_network.fit(X_train, y_train)
    prediction = the_network.predict(X_test)
    matrix = confusion_matrix(y_test, prediction)
    report = classification_report(y_test, prediction)
    report2 = classification_report(y_test, prediction, output_dict=True)

    if display_results:
        # Get Predictions
        print("Calculating Predictions \n")
        prediction = the_network.predict(X_test)

        # Get Confusion Matrix
        print("Generating Matrix \n")
        matrix = confusion_matrix(y_test, prediction)

        # Get Classification Report
        print("Compiling Report \n")
        report = classification_report(y_test, prediction)
        report2 = classification_report(y_test, prediction, output_dict=True)

        # Print Results
        print(prediction, "\n")
        print(matrix, "\n")
        print(report, "\n")

    # print("Here is report2[weighted avg][precision]: ", type(report2["weighted avg"]["precision"]))

    weighted_score = round(report2["weighted avg"]["precision"], 2)

    return the_network, weighted_score


if __name__ == "__main__":
    # Set Options for  script
    #     separated_pitch_data: this will take the data and seaparate it by pitcher
    #     get_initial_data: this will load the large initial file for cleaning, not needed if just training networks
    #     encode_data: this will encode the categorical columns of the dataset
    #     replace_nan: deprecated and should just use the Remove_Nan_Rows.py script
    #     run_Neural_Network: this will create and test a neural network, specify the parameters in the function
    #     specify_columns: this will allow you to specify which columns you want loaded into the dataframe from the csv
    #     run_RFE: this will run RFE, which can be used to rank the columns importance
    #     test_run: this will cause some functions to behave differently

    separated_pitch_data = True
    get_initial_data = True
    encode_data = True
    replace_nan = False
    run_Neural_Network = False
    specify_columns = False
    run_RFE = False
    test_run = False
    encoder = LabelEncoder()
    data_scaler = StandardScaler()

    # Specify the input CSV
    input_file = "Modeling_Data_New.csv"
    test_file = "Data/430935.csv"
    nan_columns = []

    # Load the CSV to a Pandas Dataframe
    if get_initial_data:
        # Specify the columns to load, this speeds up processing and reduces memory usage
        if specify_columns:
            columns_to_load = pd.read_csv("Columns_To_Use.csv", sep=",")
            print("Loading Data \n")
            loaded_csv = pd.read_csv(input_file, usecols=columns_to_load, sep=",", header=0)
        else:
            # Load the .csv file into a pandas dataframe
            print("Loading Data \n")
            loaded_csv = pd.read_csv(input_file, sep=",", header=0)
        print("Data finished loading\n")
        # A list of the different types of pitches, is used in a method later
        all_pitch_types = list(loaded_csv.pitch_type.unique())
        print()

        # USE Remove_Nan_Rows.py to remove the rows with nans in them
        # if replace_nan:
        #     try:
        #         for column in loaded_csv:
        #             if loaded_csv[column].isnull().values.any():
        #                 most_used_value = loaded_csv.loc[:, column].mode().values[0]
        #                 Replace_Value_in_Column(np.nan, most_used_value, column, loaded_csv)
        #     except Exception as err:
        #         print("Failure during nan replacement: ", err)


        # Encode the categorical data into numerical
        if encode_data:
            try:
                categorical_columns = ["stand", "outcomelag1", "outcomelag2", "outcomelag3",
                                       "pitch_typelag1", "pitch_typelag2", "pitch_typelag3"]
                for cat_col in categorical_columns:
                    most_used_value = loaded_csv.loc[:, cat_col].mode().values[0]
                    Replace_Value_in_Column(np.nan, most_used_value, cat_col, loaded_csv)
                    Encode_Categorical_Data(loaded_csv, cat_col, encoder)
            except Exception as ex:
                print("Failed encoding of data ", ex)

        # Separate the data by pitchers
        if separated_pitch_data:
            Separate_Pitchers(shuffle(loaded_csv), "Data/", keep_pitcher_column=False, test_run_sep_pitch=test_run)

    # Generate and test a neural network
    if run_Neural_Network:
        directory = "Data_New/*.csv"
        # Get all the files in the directory that end in .csv
        # files = glob.glob(directory)

        # This was used to specify a specific file to run the NN on
        model_name = "448306"
        test_file = "Data_New/{}.csv".format(model_name)

        # Define the layers in the NN
        layer_one = 128
        layer_two = 128
        layer_three = 0
        layer_four = 0
        layer_five = 0
        max_iterations = 100
        test_network, weighted_score = Generate_Neural_Network(filename=test_file, split_percentage=0.3,
                                                               data_scaler=data_scaler, layer_one_size=layer_one,
                                                               layer_two_size=layer_two, layer_three_size=layer_three,
                                                               layer_four_size=layer_four, layer_five_size=layer_five,
                                                               max_iterations=max_iterations, display_results=True)

        # This saves the NN and allows it to be accessed later
        Save_Network(test_network, network_name=model_name,  max_iter=max_iterations, score=weighted_score,
                     l1=layer_one, l2=layer_two, l3=layer_three, l4=layer_four, l5=layer_five)

        # Can be used to generated trained models or find most accurate variations
        # Improve_Network()

    # Can be used to rank the columns by importance.
    if run_RFE:
        current_dataframe = pd.read_csv("Data/461829.csv", sep=",", header=0, low_memory=False)

        print("X has been initialized: \n")
        X = current_dataframe.drop("pitch_type", axis=1)
        list_col = list(X)
        print("y has been initialized: \n")
        y = current_dataframe["pitch_type"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        logReg = LogisticRegression()
        rfe = RFE(logReg, 20)
        rfe.fit(X_train, y_train)

        print(rfe.support_)
        rankList = rfe.ranking_
        print(rankList)
        combined = zip(list_col, rankList)
        print("Here is combined: ", combined)
        for val in combined:
            print(val)

    print('Finished')
