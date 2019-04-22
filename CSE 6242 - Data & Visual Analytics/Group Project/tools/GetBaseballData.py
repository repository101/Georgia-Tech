# CSE-6242 Data and Visual Analytics
# Dr. Duen Horng(Polo) Chau
# Group Project - Baseball Pitch Prediction
# Josh Adams

import csv
import time
import requests
import pandas as pd
from dateutil import parser


def get_data_for_games(game_schedule_data_frame):
    # Count is used as a tracker so I can sleep(wait_time) often, as to not overload the server
    count = 0
    # date_tracker is used to record previously accessed dates so we do not process the same
    #   data multiple times
    date_tracker = {}
    wait_time = 3

    dates = game_schedule_data_frame['Date']
    try:
        # Create a session so we are able to access the site and get the .csv
        with requests.Session() as sess:
            for entry in dates:
                # Checks to see if we have processed that date, if so we continue so we do
                #   not do more work than what is needed
                if entry in date_tracker:
                    continue
                else:
                    date_tracker[entry] = True
                    if count % 10 == 1:
                        time.sleep(wait_time)
                        print('Waiting for {} seconds'.format(wait_time))
                    date_from_string = parser.parse(entry)
                    # Both start and end are the same, I am breaking this up for the .format to make it more readable
                    #   vs. just having the same variable in there multiple times
                    start_year, end_year, season_year = date_from_string.year, date_from_string.year, date_from_string.year
                    start_month, end_month = date_from_string.month, date_from_string.month
                    start_day, end_day = date_from_string.day, date_from_string.day

                    # Creation of the URL to get the .csv files
                    url_to_download_game_data = "https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfBBT=&hf" \
                                   "PR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=R%7C&hfC=&hfSea={}%7C&hfSit=&player_type" \
                                   "=pitcher&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={}-{}-{}&" \
                                   "game_date_lt={}-{}-{}&hfInfield=&team=&position=&hfOutfield=&hfRO=&home_road=&hfFlag=&hf" \
                                   "Pull=&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name&sort_col=pitches&player_" \
                                   "event_sort=h_launch_speed&sort_order=desc&min_pas=0&type=details&"\
                        .format(season_year, start_year, start_month, start_day, end_year, end_month, end_day)

                    count += 1

                    # Process the created URL
                    downloaded_data = sess.get(url_to_download_game_data)
                    decoded_content = downloaded_data.content.decode('utf-8')
                    # We needed to initialize a csv_reader as to be able to convert the decoded_content into a .csv
                    csv_reader = csv.reader(decoded_content.splitlines(), delimiter=',')

                    # Call function to process date into a .csv file
                    write_to_csv(csv_reader, start_year, start_month, start_day)

    except Exception as err_get_games:
        print("An error occurred while attempted to "
              "retrieve the data from the website. \n{}".format(err_get_games))


def write_to_csv(reader, start_year, start_month, start_day):

    try:
        # Defining the file_name schema
        file_name = "CSV\CSV_{}\Pitch_Data_CSV_{}_{}_{}.csv".format(start_year, start_year, start_month, start_day)
        with open(file_name, 'w', newline='') as output_file:
            # Initializing a csv_writer and its schema, such as having quotes
            csv_writer = csv.writer(output_file, delimiter=',', quoting=csv.QUOTE_ALL)
            for pitch_data in reader:
                # Writing the data to the file
                csv_writer.writerow(pitch_data)
        # Close the file
        output_file.close()
        print("Writing is Complete and {}.csv is closed".format(file_name))
    except EnvironmentError as err:
        print("An error occurred while trying to write the data to the csv file")
        print("\n", err)


if __name__ == "__main__":

    # Load previous CSV data
    #   2018BaseballGames.csv
    game_csv = "2008-2018_Baseball_Games.csv"
    try:
        loaded_csv = pd.read_csv(game_csv, sep=',', header=0)
        print()
        get_data_for_games(loaded_csv)
    except Exception as err_loading_game_schedule:
        print("An error occurred while trying to read the Game Schedule CSV \n {}".format(err_loading_game_schedule))

    print("FINISHED")
    print()
