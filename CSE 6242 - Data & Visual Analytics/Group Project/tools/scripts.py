import argparse
import csv
import glob
import os
import re
import time
from io import StringIO
from multiprocessing.pool import ThreadPool

import pandas as pd
import requests
from bs4 import BeautifulSoup as bs

base_dir = 'All_Pitch_Data'
raw_data_dir = base_dir + '/Raw_Data'
cleaned_data_dir = base_dir + '/Cleaned_Data'
game_schedule_dir = base_dir + '/Game_Schedule'
misc_dir = base_dir + "/Misc"


def convert_month(month, month_dict):
    # Convert the value to a string and and a 0 to the beginning if it is
    #   only a single digit. This will allow for our two digit month requirement
    if len(str(month_dict[month])) == 1:
        return "0" + str(month_dict[month])
    else:
        return str(month_dict[month])


def remove_empty_files(final_pass=False):
    filename = raw_data_dir + "/*.csv"
    directory = glob.glob(filename)
    new_dataframe = pd.DataFrame(columns=["pitcher_name", "pitcher_id", "year"])

    count = 0
    for file in directory:
        size = os.path.getsize(file)
        if size <= 993:
            split_string_one = file.split("_")
            split_string_two = split_string_one[3].split("\\")
            pitcher_id = int(split_string_one[4])
            pitcher_name = split_string_two[1][:-4]
            year = int(split_string_two[1][-4:])
            count += 1
            new_dataframe.loc[count] = [pitcher_name, pitcher_id, year]
            os.remove(file)
            if (count % 1000 == 0) and count != 0 :
                print("The current number of entries in the dataframe: \n", len(new_dataframe['pitcher_name']))
    if not final_pass:
        new_dataframe.to_csv(misc_dir + "/Second_Pass_Pitcher_Data.csv", sep=',', index=False)
        return misc_dir + "/Second_Pass_Pitcher_Data.csv"
    else:
        new_dataframe.to_csv(misc_dir + "/Final_Pass_Pitcher_Data.csv", sep=',', index=False)
        return misc_dir + "/Final_Pass_Pitcher_Data.csv"



    return misc_dir + "/Second_Pass_Pitcher_Data.csv"


def get_list_of_pitchers():
    print("Getting list of pitchers with ID's and number of pitches thrown.\n")
    pitcher_list_url = "https://baseballsavant.mlb.com/statcast_search?hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&" \
                       "stadium=&hfBBL=&hfNewZones=&hfGT=R%7C&hfC=&hfSea=2019%7C2018%7C2017%7C2016%7C2015" \
                       "%7C2014%7C2013%7C2012%7C2011%7C2010%7C2009%7C2008%7C2007%7C2006%7C2005%7C2004%7C2003" \
                       "%7C2002%7C2001%7C2000%7C&hfSit=&player_type=pitcher&hfOuts=&opponent=&pitcher_throws=" \
                       "&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfInfield=&team=&position=&hfOutfield=" \
                       "&hfRO=&home_road=&hfFlag=&hfPull=&metric_1=&hfInn=&min_pitches=1000&min_results=0&group_" \
                       "by=name&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc&min_pas=0"
    page_response = requests.get(pitcher_list_url, timeout=40)
    parsed_html = bs(page_response.content, 'html.parser')  # Parse the page response as html

    re_find_id = re.compile("[0-9]{6}")
    re_find_numbers = re.compile("(\d+)")
    columns = ["pitcher_name", "pitcher_id", "pitches_thrown"]
    the_dataframe = pd.DataFrame(columns=columns)

    html_player_data = parsed_html.find_all("tr", {"class": "search_row"})

    for i in range(len(html_player_data)):
        pitcher_info = html_player_data[i].find_all("td", {"class": "player_name"})
        pitcher_thrown = html_player_data[i].find_all("td", {"class": "numeric"})
        pitcher_name = pitcher_info[0].text
        pitches_thrown = re_find_numbers.findall(str(pitcher_thrown[0].text))
        pitcher_id = re_find_id.findall(str(pitcher_info[0]))
        the_dataframe.loc[i] = [pitcher_name, pitcher_id[0], pitches_thrown[0]]

    the_dataframe.to_csv(misc_dir + "/Pitcher_List.csv", sep=",", index=False)
    print("Pitcher list saved to {}.\n".format(misc_dir + "/Pitcher_List.csv"))
    return misc_dir + "/Pitcher_List.csv"


def get_schedules_csv(start_year, end_year, year_increment, month_dictionary,):
    combined_data = []  # A list to hold all of the data for the specified years
    if (start_year > 2018) or (start_year < 1990) or (start_year > end_year):
        print("\nThere was a problem with your start year\n")
        return False
    if (end_year > 2019) or (end_year < 1990) or (end_year < start_year):
        print("\nThere was a problem with your end year\n")
        return False

    try:

        for year in range(start_year, end_year, year_increment):
            # The URL for the baseball game schedules
            schedule_url = "https://www.baseball-reference.com/leagues/MLB/{}-schedule.shtml".format(year)
            print("Here is schedule_url: " + schedule_url)
            page_response = requests.get(schedule_url, timeout=3)
            parsed_html = bs(page_response.content, 'html.parser')  # Parse the page response as html

            # Looking at the html for the site I found that the information we are looking for
            #   is in a <div> with a class='section_content'. There is typically only two to three of these
            #   per page on the site. The first occurrence is our Regular Season, the second is Post Season,
            #   and the third is typically the section at the bottom of the screen to select different features
            html_game_data = parsed_html.find_all("div", {"class": "section_content"})
            game_schedule = {}

            # We are only going to work with the first element in html_game_data, since
            #   that corresponds to the data we are looking for.  Breaking up the
            #   aggregate due to having unnecessary data, such as individual '\n'
            broken_aggregate = html_game_data[0].find_all("div")

            count = 0
            # Now we have removed the unnecessary data we will iterate over the results to get the game data
            for result in broken_aggregate:
                temp_length = len(broken_aggregate)
                temp_percent = [int(temp_length / 4), int(temp_length / 4) * 2, int(temp_length / 4) * 3]
                final_game_list = []
                if count == temp_percent[0]:
                    print("25% complete with year {}\n".format(year))
                elif count == temp_percent[1]:
                    print("50% complete with year {}\n".format(year))
                elif count == temp_percent[2]:
                    print("75% complete with year {}\n".format(year))

                # The data is store in the 'h3' tag so we can just grab it here and associate it
                #   with each of the games in this aggregate. The reason we can do this is that they are
                #   already broken up by <div>, where each <div> corresponds to a different date.
                temp_date = result.h3.text

                # Currently the date is given as "DayOfTheWeek", "Month Day", "Year"
                #   So we can split this up to get the areas we want
                # Date[0]: Day of the Week
                # Date[1]: Month and Day
                # Date[2]: Year
                date = temp_date.split(', ')
                temp_date = date[1].split(" ")
                the_year = date[2]

                # Pulled out the conversion of the month into a separate function
                the_month = convert_month(temp_date[0], month_dictionary)
                the_day = temp_date[1]
                day_of_the_week = date[0]

                # "YYYY-MM-DD"
                date_string = the_year + "-" + the_month + "-" + the_day
                count += 1
                for game in result:
                    # Removal of unnecessary data
                    if len(game) == 1:
                        continue
                    elif ("Standings & Scores" in game.text) or ("Preview" in game.text) or (game is None):
                        continue
                    else:
                        # Regular expression to find the scores
                        re_find_scores = re.compile("(\d+)")
                        scores = re_find_scores.findall(game.text)

                        # Setting the scores for the specific teams. Home team is index 1 due to it showing
                        #   up on the website second
                        home_team_score = scores[1]
                        away_team_score = scores[0]

                        # The away team always shows up first on the website so it was easily obtained
                        away_team_name = game.find("a").text

                        # The home team shows up next so we grab the next 'a' section and split at the '\n'
                        #   because the home team information is being coupled with their score. We just split
                        #   and take the first instance
                        home_team_name = game.find("a").find_next().text.split("\n")[0]

                        # Generating the list that will ultimately be used to write out the rows in the csv
                        game_data = [home_team_name.strip(), home_team_score,
                                     away_team_name, away_team_score,
                                     "{} ({}) -- {} ({})".format(home_team_name, home_team_score,
                                                                 away_team_name, away_team_score),
                                     date_string]

                        # We append the game list because we are creating a list of games within a dictionary.
                        final_game_list.append(game_data)
                    game_schedule[date_string] = final_game_list

            # Combined data will be a list of dictionaries. The list is of the years and the dictionaries
            #   hold all the data for the specific dates within those years.
            combined_data.append(game_schedule)
        return combined_data
    except Exception as err:
        print("Something went wrong when gathering the data from the website\n")
        print("The Error: \n", err)


def write_game_schedules_to_csv(input_data, file_name):
    # Row is only used to create the header in the csv file
    row = ["HomeTeamName", "HomeTeamScore", "AwayTeamName", "AwayTeamScore", "Score", "Date"]
    try:
        # This will create a .csv file in the same directory that this python script is located
        with open("{}.csv".format(str(file_name)), 'w', newline='') as csv_file:
            print("\nOpening {}.csv".format(str(file_name)))
            # Initializing the csv writer
            writer = csv.writer(csv_file)
            # Writing the header to the csv file
            writer.writerow(row)
            print("\nWriting Data to {}.csv".format(file_name))
            # Iterating through input_data and writing those rows to the csv
            for i in input_data:
                for key in i:
                    for game in i[key]:
                        writer.writerow(game)

        csv_file.close()
        print("\nWriting is Complete and {}.csv is closed".format(file_name))
    except EnvironmentError as err:
        print("An error occurred while trying to write the data to the csv file")
        print("\n", err)


# def get_games_csv_OLD(game_schedule_data_frame, save_individual_files=False):
#     # DEPRICATED TOO SLOW
#     save_1 = True
#     print("Please be patient, this will take a VERY long time.")
#     print("We have added pauses to this script as to not overload the "
#           "website we are scraping, this will add to wait time")
#     date_tracker = {}
#     # columns_file = raw_data_dir + "/columns_to_load.csv"
#     # columns_to_load = pd.read_csv(columns_file)
#     # list_columns_to_load = list(columns_to_load)
#
#     dates = game_schedule_data_frame['Date']
#     count = 0
#     length_of_dates = len(dates)
#     percentage_list = [(i * round((length_of_dates / 100))) for i in range(1, 100)]
#     for entry in dates:
#         if entry in date_tracker:
#             continue
#         else:
#             if count in percentage_list:
#                 print("{}% complete".format((percentage_list.index(count) + 1)))
#             if (count % 20 == 0) and (count > 0):
#                 time.sleep(2)
#             date_from_string = parser.parse(entry)
#             # # Both start and end are the same, I am breaking this up for the .format to make it more readable
#             # #   vs. just having the same variable in there
#             start_year, end_year, season_year = date_from_string.year, date_from_string.year, date_from_string.year
#             start_month, end_month = date_from_string.month, date_from_string.month
#             start_day, end_day = date_from_string.day, date_from_string.day
#
#             schedule_url = "https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfBBT=&hf" \
#                            "PR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=R%7C&hfC=&hfSea={}%7C&hfSit=&player_type" \
#                            "=pitcher&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={}-{}-{}&" \
#                            "game_date_lt={}-{}-{}&hfInfield=&team=&position=&hfOutfield=&hfRO=&home_road=&hfFlag=&hf" \
#                            "Pull=&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name&sort_col=pitches&player_" \
#                            "event_sort=h_launch_speed&sort_order=desc&min_pas=0&type=details&" \
#                 .format(season_year, start_year, start_month, start_day, end_year, end_month, end_day)
#             date_tracker[entry] = schedule_url
#             count += 1
#             with requests.Session() as sess:
#                 download = sess.get(schedule_url)
#                 decoded_content = download.content.decode('utf-8')
#                 csv_reader = csv.reader(decoded_content.splitlines(), delimiter=',')
#                 if save_1:
#                     temp = pd.read_csv(StringIO(decoded_content), header=0)
#                     temp.to_csv(raw_data_dir + '/test.csv', sep=',', index=False)
#                     save_1 = False
#                     combined_dataframe = pd.read_csv(raw_data_dir + '/test.csv', sep=',', header=0)
#                     pd.DataFrame(list(combined_dataframe)).to_csv(misc_dir + "/columns_to_load.csv",
#                                                                   sep=',', index=False)
#                 else:
#                     if save_individual_files:
#                         temp_df = pd.read_csv(StringIO(decoded_content), header=0)
#                         temp_df = temp_df[list_columns_to_load]
#                         temp_df.to_csv('{}/{}.csv'.format(raw_data_dir, entry), sep=',', index=False)
#                         combined_dataframe = pd.concat([combined_dataframe, temp_df],
#                                                        ignore_index=True, axis=0, sort=False)
#                         print()
#                     else:
#                         temp_df = pd.read_csv(StringIO(decoded_content), header=0)
#                         combined_dataframe = pd.concat([combined_dataframe[list_columns_to_load],
#                                                         temp_df[list_columns_to_load]],
#                                                        ignore_index=True, axis=0, sort=False)
#     if save_individual_files:
#         return
#     else:
#         combined_dataframe.to_csv("Combined_Data.csv", sep=',', index=False)
#         return "Combined_Data.csv"


def write_game_to_csv(reader, start_year, start_month, start_day):
    try:
        file_name = "{}/Pitch_Data_CSV_{}_{}_{}.csv".format(raw_data_dir, start_year, start_month, start_day)
        with open(file_name, 'w', newline='') as output_file:
            csv_writer = csv.writer(output_file, delimiter=',', quoting=csv.QUOTE_ALL)
            for pitch_data in reader:
                csv_writer.writerow(pitch_data)
        output_file.close()
        print("Writing is Complete and {} is closed".format(file_name))
        return file_name
    except EnvironmentError as err:
        print("An error occurred while trying to write the data to the csv file")
        print("\n", err)


def clean_data(the_data):

    return the_data


def get_games_csv(pitcher_csv, second_pass=False):
    print("Please be patient, this will take a long time.")
    print("We have added pauses to this script as to not overload the "
          "website we are scraping, this will add to wait time")
    pitcher_dataframe = pd.read_csv(pitcher_csv, sep=',', header=0)
    count = 0
    URL_List = []
    for key, value in pitcher_dataframe.iterrows():
        if not second_pass:
            for year in range(2008, 2019, 1):
                pitcher_data_url = "https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfBBT=" \
                                   "&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=R%7C&hfC=&hfSea=2019%7C2018%7C2017" \
                                   "%7C2016%7C2015%7C2014%7C2013%7C2012%7C2011%7C2010%7C2009%7C2008%7C&hfSit=&" \
                                   "player_type=pitcher&hfOuts=" \
                                   "&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={}-02-01&game_" \
                                   "date_lt={}-02-01&hfInfield=&team=&position=&hfOutfield=&hfRO=&home_road=&hfFlag=" \
                                   "&hfPull=&pitchers_lookup%5B%5D={}&metric_1=&hfInn=&min_pitches=1000&min_results=0" \
                                   "&group_by=name&sort_col=pitches&player_event_sort=h_launch_speed&sort_order" \
                                   "=desc&min_pas=0&type=details&".format(year, (year+1), value[1])
                URL_List.append([pitcher_data_url, (value[0] + str(year)), value[1]])
        else:
            pitcher_data_url = "https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfBBT=" \
                               "&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=R%7C&hfC=&hfSea=2019%7C2018%7C2017" \
                               "%7C2016%7C2015%7C2014%7C2013%7C2012%7C2011%7C2010%7C2009%7C2008%7C&hfSit=&" \
                               "player_type=pitcher&hfOuts=" \
                               "&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={}-02-01&game_" \
                               "date_lt={}-02-01&hfInfield=&team=&position=&hfOutfield=&hfRO=&home_road=&hfFlag=" \
                               "&hfPull=&pitchers_lookup%5B%5D={}&metric_1=&hfInn=&min_pitches=1000&min_results=0" \
                               "&group_by=name&sort_col=pitches&player_event_sort=h_launch_speed&sort_order" \
                               "=desc&min_pas=0&type=details&".format(value[2], value[2], value[0])
            URL_List.append([pitcher_data_url, (value[0]+str(value[2])), value[1]])

        count += 1
    percent_list = [(j * round(len(URL_List)/100)) for j in range(100)]
    percent_set = set(percent_list)

    print("Starting to grab data.\n")
    for i in range(0, (len(URL_List) + 1), 5):
        temp = set([i for i in range(i-3, i+2, 1)])
        set_intersect = percent_set.intersection(temp)
        if len(set_intersect) > 0:
            temp = set_intersect.pop()
            if temp != 0:
                print("{}% Completed".format((percent_list.index(temp)) + 1))
                time.sleep(10)
        try:
            if not second_pass:
                result = ThreadPool(5).imap_unordered(get_session, URL_List[i:i+5])
                result.close()
                result.join()
            else:
                result = ThreadPool(2).imap_unordered(get_session, URL_List[i:i+5])
                result.close()
                result.join()

        except Exception as err:
            continue

    return True

def combine_csv(directory):
    print("Processing csv files so we can combine them.\n")
    all_files = glob.glob(directory + "/*.csv")
    number_of_files = len(all_files)
    percent_list = [(i * round((number_of_files/100))) for i in range(100)]
    test = []
    for num in range(number_of_files - 1):
        if (num in percent_list) and (num != 0):
            print("{}% Completed".format(percent_list.index(num)))
        temp = pd.read_csv(all_files[num], sep=',', header=0)
        test.append(temp)
    print("Attempting to combine all csv files into one.\n")
    combined_dataframe = pd.concat(test, ignore_index=True)

    combined_dataframe.to_csv(raw_data_dir + "/Combined_Data.csv", sep=',', index=False)
    print("Finished saving the combines csv file to {}.\n".format(raw_data_dir + "/Combined_Data.csv"))
    return


def get_session(url):
    the_url, name, the_id = url
    if not os.path.exists("{}/{}_{}_pitch_data.csv".format(raw_data_dir, name, the_id)):
        try:
            r = requests.session().get(the_url)
            decoded_content = r.content.decode('utf-8')
            temp = pd.read_csv(StringIO(decoded_content), header=0)
            temp.to_csv("{}/{}_{}_pitch_data.csv".format(raw_data_dir, name, the_id), sep=',', index=False)
        except Exception as errr:
            pass
    return


def into_function():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-start", "--start_year", required=False,
                    help="start year of your search")
    ap.add_argument("-end", "--end_year", required=False,
                    help="end year of your search")
    args = vars(ap.parse_args())
    print("The first thing we need to do is get a record of all baseball games played throughout a given year.\n")
    temp_start = ''
    temp_end = ''
    if args["start_year"] is None:
        while (len(temp_start) == 0) or (not temp_start.isnumeric()):
            start = input("Which year would you like to start gathering game schedules for? (eg. 2015)\n")
            temp_start = start
    else:
        start = args["start_year"]

    if args["end_year"] is None:
        while (len(temp_end) == 0) or (not temp_end.isnumeric()):
            end = input("Which year would you like to stop gathering game schedules for? (eg. 2019)\n")
            temp_end = end
    else:
        end = args["end_year"]

    print(
        "We will begin gathering game schedules starting from year {} and until, but not including {}.\n".format(start,
                                                                                                                 end))

    # Create some folders to hold data
    if not os.path.exists(base_dir):
        # To hold All data
        os.makedirs(base_dir)

    if not os.path.exists(raw_data_dir):
        # To hold raw data from the website scraping
        os.makedirs(raw_data_dir)

    if not os.path.exists(cleaned_data_dir):
        # To hold cleaned data
        os.makedirs(cleaned_data_dir)

    if not os.path.exists(game_schedule_dir):
        # To hold game schedules
        os.makedirs(game_schedule_dir)

    if not os.path.exists(misc_dir):
        # To hold misc files
        os.makedirs(misc_dir)

    temp_schedule = ''
    while (len(temp_schedule) == 0) or (temp_schedule.lower() not in ['yes', 'y', 'no', 'n']):
        get_game_schedule = input("Do you want to get the game schedules? (eg. Yes or No)\n")
        temp_schedule = get_game_schedule
    if get_game_schedule.lower() in ['y', 'yes']:
        get_game_schedule = True
    else:
        get_game_schedule = False

    if get_game_schedule:
        temp_get_game_schedule = ''
        while (len(temp_get_game_schedule) == 0) or (temp_get_game_schedule.lower() not in ['yes', 'y', 'no', 'n']):
            ONLY_GET_GAME_SCHEDULES = input("Do you want to get just game schedules? (eg. Yes or No)\n")
            temp_get_game_schedule = ONLY_GET_GAME_SCHEDULES

        if ONLY_GET_GAME_SCHEDULES.lower() in ['yes', 'y']:
            ONLY_GET_GAME_SCHEDULES = True
        else:
            ONLY_GET_GAME_SCHEDULES = False
    else:
        ONLY_GET_GAME_SCHEDULES = False

    return ONLY_GET_GAME_SCHEDULES, start, end

