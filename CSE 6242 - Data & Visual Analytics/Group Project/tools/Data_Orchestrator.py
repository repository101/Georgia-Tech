# CSE-6242 Data and Visual Analytics
# Dr. Duen Horng(Polo) Chau
# Group Project - Baseball Pitch Prediction using Recurrent Neural Networks
# Josh Adams

import sys

from scripts import *

if __name__ == "__main__":
    ONLY_GET_GAME_SCHEDULES, start, end = into_function()

    # A dictionary to convert the month into its numerical value
    months = {"January": 1, "February": 2, "March": 3, "April": 4,
              "May": 5, "June": 6, "July": 7, "August": 8,
              "September": 9, "October": 10, "November": 11, "December": 12}
    pitcher_list_filename = get_list_of_pitchers()

    # Combined_data will hold all the data
    cleaned_schedule_data = get_schedules_csv(int(start), int(end), 1, months,)
    schedule_filename = "{}/{}_Baseball_Games".format(game_schedule_dir, (str(start) + "-" + str(int(end) - 1)))

    # Create a csv file of game schedules with the specified name
    if cleaned_schedule_data != False:
        write_game_schedules_to_csv(cleaned_schedule_data, schedule_filename)

    if ONLY_GET_GAME_SCHEDULES:
        print("\nOnly the game schedules were requested and that task is complete.")
        sys.exit()

    # Beginning to gather pitch data for each game
    loaded_csv = pd.read_csv(pitcher_list_filename, sep=',', header=0)
    get_games_csv(pitcher_list_filename)
    second_pass_csv = remove_empty_files(final_pass=False)
    get_games_csv(second_pass_csv, second_pass=True)
    final_pass_csv = remove_empty_files(final_pass=True)
    combine_csv(raw_data_dir)
    print("FINISHED!!!\n")
