# CSE-6242 Data and Visual Analytics
# Dr. Duen Horng(Polo) Chau
# Group Project - Baseball Pitch Prediction
# Josh Adams

# the url = https://www.baseball-reference.com/leagues/MLB/2018-schedule.shtml

# Each date is separated in the HTML in <div class="section_content" id="div_9510680018">
#   Inside the <div class="section_content" id="div_9510680018"> each date for scheduled games
#       is separated in a separate <div></div>
#           Inside each <div></div> the <h3> tag is the date in the form Thursday, March 29, 2018
#           Each is in a <p class="game">
#           Each team is in a <a href="somewebsite.html"> TEAM NAME</a>


import re
import csv
import argparse
import requests
from bs4 import BeautifulSoup as bs


def convert_month(month, month_dict):

    # Convert the value to a string and and a 0 to the beginning if it is
    #   only a single digit. This will allow for our two digit month requirement
    if len(str(month_dict[month])) == 1:
        return "0" + str(month_dict[month])
    else:
        return str(month_dict[month])


def get_games_csv(start_year, end_year, year_increment, month_dictionary):

    combined_data = []  # A list to hold all of the data for the specified years
    if(start_year > 2018) or(start_year < 1990) or(start_year > end_year):
        print("\nThere was a problem with your start year\n")
        return False
    if(end_year > 2019) or (end_year < 1990) or (end_year < start_year):
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

            # Now we have removed the unnecessary data we will iterate over the results to get the game data
            for result in broken_aggregate:
                final_game_list = []

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


def write_to_csv(input_data, file_name):
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


if (__name__ == "__main__"):
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-start", "--start_year", required=False,
                    help="start year of your search")
    ap.add_argument("-end", "--end_year", required=False,
                    help="end year of your search")
    args = vars(ap.parse_args())

    if(args["start_year"] is None):
        start = 2015
    else:
        start = args["start_year"]

    if(args["end_year"] is None):
        end = 2019
    else:
        end = args["end_year"]

    # A dictionary to convert the month into its numerical value
    months = {"January": 1, "February": 2, "March": 3, "April": 4,
                  "May": 5, "June": 6, "July": 7, "August": 8,
                  "September": 9, "October": 10, "November": 11, "December": 12}

    print("\nStart: {}    End: {}".format(start, end))

    # Combined_data will hold all the data
    cleaned_data = get_games_csv(int(start), int(end), 1, months)

    # Create a csv file with the specified name
    if (cleaned_data != False):
        write_to_csv(cleaned_data, "{}_Baseball_Games".format(str(start)+"-"+str(int(end)-1)))
