import http.client
import re
import csv
import time
import sys


def findFirstSetOfMovieData(apiKey):
    # Setting up a connection
    connection_request = http.client.HTTPSConnection("api.themoviedb.org")
    # Defining a dictionary to hold the results   results_dict[MOVIE_ID]: MOVIE_TITLE
    movie_dict = {}
    movie_ids = []
    web_page = 0
    while len(movie_dict) < 350:
        api_url = 'https://api.themoviedb.org/3/discover/movie?api_key={}&language=en-US&sort_by=popularity.desc&include_adult=false&include_video=false&page={}&primary_release_date.gte=2004&with_genres=18'.format(
            apiKey, web_page)
        payload = "{}"
        connection_request.request("GET", api_url, payload)
        connection_response = connection_request.getresponse()
        raw_data_from_api = connection_response.read()
        decoded_data_from_api = raw_data_from_api.decode('UTF-8')

        # Using Regex to search for all of the titles in decoded_data_from_api
        tempTitles = re.findall('"title":"(.*?)"', decoded_data_from_api)

        # Using Regex to search for all of the ids in decoded_data_from_api
        tempIds = re.findall('"id":([0-9]+),', decoded_data_from_api)
        movie_ids += tempIds
        web_page += 1
        for entry in range(len(tempTitles)):
            if (len(movie_dict) >= 350):
                continue
            else:
                movie_dict[tempIds[entry]] = tempTitles[entry]
    return movie_dict, movie_ids[:350]


def findSecondSetOfMovieData(api_key, movie_ids, first_set_of_movie_data):
    # Setting up second connection
    connection_1 = http.client.HTTPSConnection("api.themoviedb.org")

    # Defining a dictionary to hold the results   results_dict[MOVIE_ID]: SIMILAR_MOVIE_ID
    results_list = []

    for temporary_tuple in range(len(first_set_of_movie_data)):
        # Used to prevent API timeout due to 40 calls per 10 second restriction
        if (temporary_tuple % 40 == 0) and (temporary_tuple != 0):
            print(
                "Wait 10 seconds due to restriction on the number of API calls we are able to"
                " make in a certain time frame\n")
            time.sleep(10)

        # API Call to get similar movies to the passed in movie_id
        api_url1 = 'https://api.themoviedb.org/3/movie/{}/similar?page=1&language=en-US&api_key={}'.format(movie_ids[temporary_tuple],
                                                                                                           api_key)
        payload1 = "{}"
        connection_1.request("GET", api_url1, payload1)
        connection_response = connection_1.getresponse()
        raw_data_from_api = connection_response.read()
        decoded_data_from_api = raw_data_from_api.decode('UTF-8')

        # Using Regex to search for all of the titles in decoded_data_from_api
        temporary_Title_list = re.findall('"title":"(.*?)"', decoded_data_from_api)
        for j in temporary_Title_list:
            # Used to clean data as '\\u0026' and '\u0394' we both causing issues
            if '\\u0026' in j:
                temporary_Title_list[temporary_Title_list.index(j)] = j.replace('\\u0026', 'And')
            if '\u0394' in j:
                temporary_Title_list[temporary_Title_list.index(j)] = j.replace('\u0394', '{DELTA SYMBOL}')

        # Using Regex to search for all of the ids in decoded_data_from_api
        temporary_Id_list = re.findall('"id":([0-9]+),', decoded_data_from_api)

        # Since we want at most 5 result, if the length of our results is greater
        #    than 5 we will take at most 5 results. Otherwise we will just use
        #    the results returned
        if len(temporary_Title_list) > 5:
            temporary_Title_list = temporary_Title_list[:5]
        if len(temporary_Id_list) > 5:
            temporary_Id_list = temporary_Id_list[:5]
        for index in temporary_Id_list:
            results_list.append([movie_ids[temporary_tuple], index])

    # Converting from a list of lists to a list of tuples
    results_list_to_tuples = []
    for information in results_list:
        results_list_to_tuples.append(tuple(information))

    # Converting the list of tuples into a set, to remove duplicates
    set_of_tuples = set(results_list_to_tuples)
    tuples_to_remove = []

    # Iterating over the set of tuples to find any other duplicates because in
    #    our case A:B is the same as B:A. We only want to keep the occurrence where
    #    [movie] < [similar_movie_id]
    for temporary_tuple in set_of_tuples:
        temp = [temporary_tuple[1], temporary_tuple[0]]
        testTuple = tuple(temp)
        if testTuple in set_of_tuples:
            if temporary_tuple[0] < temporary_tuple[1]:
                tuples_to_remove.append(testTuple)
            else:
                tuples_to_remove.append(temporary_tuple)

    # Converting tuples_to_remove to a set to remove duplicates
    tuples_to_remove = set(tuples_to_remove)

    # Iterating over tuples_to_remove and removing that occurrence of the tuple from set_of_tuples
    for tuple_to_remove in tuples_to_remove:
        set_of_tuples.remove(tuple_to_remove)

    return set_of_tuples


def writeDataToCSV(data_file, file_name, file_type):
    # A method to take a data_file(typically a dictionary) and a file_name and
    #     create a .csv file from the passed in data_file
    with open(file_name, 'w', encoding='UTF-8') as csv_file:
        field_names = ['movie-ID', 'movie-name']
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        if (file_type == 'dictionary') or (file_type == 'Dictionary'):
            for data in data_file:
                writer.writerow({'movie-ID': data, 'movie-name': data_file[data]})
        elif (file_type == 'set') or (file_type == 'Set'):
            # Change the way we process the file due to the way Sets can be accessed
            for data in data_file:
                writer.writerow({'movie-ID': data[0], 'movie-name': data[1]})
        else:
            return 'You did not pass in a valid file type: (Dictionary/Set)'

def main():
    # Call findFirstSetOfMovieData with the passed in API_Key from the command line.
    #    This method will return a dictionary containing the movie data as well as a
    #    list of the movie ids found so the process of finding similar movies will be easier
    firstSetOfMovieData, movie_ids = findFirstSetOfMovieData(sys.argv[1])

    # Write firstSetOfMovieData to a .csv file named 'movie_ID_name.csv'
    writeDataToCSV(firstSetOfMovieData, 'movie_ID_name.csv', 'Dictionary')

    # Call findSecondSetOfMovieData with passed in API_Key from the command line.
    #    This method will return a dictionary containing the movie data.
    secondSetOfMovieData = findSecondSetOfMovieData(sys.argv[1], movie_ids, firstSetOfMovieData)

    # Write secondSetOfMovieData to a .csv file name 'movie_ID_sim_movie_ID.csv
    writeDataToCSV(secondSetOfMovieData, 'movie_ID_sim_movie_ID.csv', 'Set')

main()
