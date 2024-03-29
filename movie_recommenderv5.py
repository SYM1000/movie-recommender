import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import os
# save numpy array as npy file
from numpy import load
from numpy import save
import numpy as np

'''
This function takes one movie as argument and returns all the recommended movies(>=50) as dictionary.
This fucntion response time is aprox: 0.25s.
With this function when can change the values of the basic parameter, since we compute them every time
'''
def get_recommendation_server(liked_movie):
    total_time = time.time()
    start_time = time.time()

    # Get constant values from npy file
    constant_all_movie_titles = load('all_movie_titles.npy', allow_pickle=True)
    # print("--- sacar columan de peliculas tarda %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    # Get cosine values from npy file
    constant_cosine_sim_keywords = load('cosine_sim_keywords.npy', allow_pickle=True)
    constant_cosine_sim_cast = load('cosine_sim_cast.npy', allow_pickle=True)
    constant_cosine_sim_genres = load('cosine_sim_genres.npy', allow_pickle=True)
    constant_cosine_sim_director = load('cosine_sim_director.npy', allow_pickle=True)
    # print("--- sacar el cosine similarity tarda %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    np_df = load('np_dataframe.npy', allow_pickle=True)

    df = pd.DataFrame(np_df, columns = ['index','budget','genres', 'homepage', 'id', 'keywords', 'original_language', 'original_title', 'overview', 'popularity', 'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime', 'spoken_languages', 'status', 'tagline', 'title', 'vote_average', 'vote_count', 'cast', 'crew', 'director', 'other'])
    # print("--- leer el dataframe tarda %s seconds ---" % (time.time() - start_time))

    try:
        start_time = time.time()
        movie_index = get_index_from_title(liked_movie, df) #Gustos del usuario
        # print("--- sacar el index tarda %s seconds ---" % (time.time() - start_time))
    except:
        # Return false if movie was not found
        print("Movie not found")
        return False

    start_time = time.time()
    similar_movies_keywords = list(enumerate(constant_cosine_sim_keywords[movie_index]))
    similar_movies_cast = list(enumerate(constant_cosine_sim_cast[movie_index]))
    similar_movies_genres = list(enumerate(constant_cosine_sim_genres[movie_index]))
    similar_movies_director = list(enumerate(constant_cosine_sim_director[movie_index]))
    # print("--- sacar los similar %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    final_scores = get_final_scores_server(constant_all_movie_titles, similar_movies_keywords, similar_movies_cast, similar_movies_genres, similar_movies_director)
    # print("--- sacar el score final %s seconds ---" % (time.time() - start_time))

    
    start_time = time.time()
    final_scores.sort(key=lambda x:x[1], reverse=True)
    # final_scores = final_scores[0:10] # Get first 10 elements
    # print("--- ordenamiento tardo %s seconds ---" % (time.time() - start_time))

    # using dictionary comprehension to convert lists to dictionary
    start_time = time.time()
    recommendation_dict = {final_scores[i][0]: final_scores[i][1] for i in range(len(final_scores)) }
    # print("--- Convertir la lista a diccionario tardao %s seconds ---" % (time.time() - start_time))

    # print("--- Recommendations computed in %s seconds ---" % (time.time() - total_time))
    return recommendation_dict

'''
This function works great when we have more than 1 liked movie. The response time is 0.35.
This function uses a npy file with the recommendations previously computed. There is no variation on the parameters(keywords, genders, cast, director)
This function takes a dictionary of liked movies and returns a dictionary with the recommendations with their value
'''
def get_recommendations_from_npy_file(liked_movies):
    total_time = time.time()
    recommendation_movies = {}

    known_recommendations = np.load("recommendations_dictionary.npy",allow_pickle=True)

    for liked in liked_movies:
        recommendations_for_one_movie = known_recommendations.item().get(liked)

        for movie in recommendations_for_one_movie:
            if movie not in liked_movies and movie not in recommendation_movies:
                recommendation_movies[movie] = recommendations_for_one_movie[movie]

    # Sort the results
    recommendation_movies = dict(sorted(recommendation_movies.items(), key=lambda item: item[1], reverse=True))

    print("--- Recommendations for many movies computed in %s seconds ---" % (time.time() - total_time))
    return recommendation_movies

'''
Not recommended
'''
def get_recommendation_from_list_server(liked_movies):
    total_time = time.time()
    recommendation_movies = {}

    for liked in liked_movies:
        recommendations = get_recommendation_server(liked)

        for movie in recommendations:
            if movie not in liked_movies and  movie not in recommendation_movies:
                recommendation_movies[movie] = recommendations[movie]

    # Sort the results
    recommendation_movies = dict(sorted(recommendation_movies.items(), key=lambda item: item[1], reverse=True))

    print("--- Recommendations for many movies computed in %s seconds ---" % (time.time() - total_time))
    return recommendation_movies

# HELPER FUNCTIONS
def get_final_scores_server(all_movies, similar_movies_keywords, similar_movies_cast, similar_movies_genres, similar_movies_director):
    final_scores = []
    # Importancias con buenos resultados: keywords(0.00), cast(0.20), genres(0.60), director(0.20)
    importancia_keywords = 0.0
    importancia_cast = 0.20
    importancia_genero = 0.60
    importancia_director = 0.20
    # Nota: la importancia en el futuro puede ser dinamica y calculada automaticamente en base a las peliculas favoritas del usuario

    i = 0
    for movie in similar_movies_keywords:

        index = movie[0]

        keyword_score = similar_movies_keywords[i][1]
        cast_score = similar_movies_cast[i][1]
        genres_score = similar_movies_genres[i][1]
        director_score = similar_movies_director[i][1]

        final_score = ( (keyword_score * importancia_keywords) + (cast_score * importancia_cast) + (genres_score * importancia_genero) + (director_score * importancia_director) )
        final_score = round(final_score*100,0)

        if final_score < 50 or final_score == 100: #For getting the scores grather than 50
            i+=1
            continue

        new = (all_movies[i], final_score)
        final_scores.append(new)
        i+=1

    return final_scores

def get_recommendation(liked_movie, cosine_sim_keywords, cosine_sim_cast, cosine_sim_genres, cosine_sim_director):

    try:
        movie_index = get_index_from_title(liked_movie, df) #Gustos del usuario
    except:
        # Return false if movie was not found
        return False

    similar_movies_keywords = list(enumerate(cosine_sim_keywords[movie_index]))
    similar_movies_cast = list(enumerate(cosine_sim_cast[movie_index]))
    similar_movies_genres = list(enumerate(cosine_sim_genres[movie_index]))
    similar_movies_director = list(enumerate(cosine_sim_director[movie_index]))

    final_scores = get_aray_of_tuples_with_final_scores(similar_movies_keywords, similar_movies_cast, similar_movies_genres, similar_movies_director)

    # sorted_final_scores = sorted(final_scores, key=lambda x:x[1], reverse= True) # Peliculas ordernadas de mayor a menor gusto

    return final_scores

# HELPER FUNCTIONS
def combine_features(row):
    try:
        return row['keywords'] +" "+ row['cast']+" "+row["genres"]+" "+row["director"]
    except:
        print("Error", row)

def get_title_from_index(index, dataFrame):
    return dataFrame[dataFrame.index == index]["title"].values[0]

def get_index_from_title(title, dataFrame):
    return dataFrame[dataFrame.title == title]["index"].values[0]

def get_array_of_values(similar_movies2):
    new_array = []

    for x in similar_movies2:
        v = x[1]
        new_array.append(v)

    return new_array

def get_aray_of_tuples_with_final_scores(similar_movies_keywords, similar_movies_cast, similar_movies_genres, similar_movies_director):
    final_scores = []
    # Importancias con buenos resultados: keywords(0.00), cast(0.20), genres(0.60), director(0.20)
    importancia_keywords = 0.25
    importancia_cast = 0.25
    importancia_genero = 0.25
    importancia_director = 0.25
    # Nota: la importancia en el futuro puede ser dinamica y calculada automaticamente en las peliculas favoritas del usuario

    i = 0
    for movie in similar_movies_keywords:

        index = movie[0]

        keyword_score = similar_movies_keywords[i][1]
        cast_score = similar_movies_cast[i][1]
        genres_score = similar_movies_genres[i][1]
        director_score = similar_movies_director[i][1]

        final_score = ( (keyword_score * importancia_keywords) + (cast_score * importancia_cast) + (genres_score * importancia_genero) + (director_score * importancia_director) )
        new = (index, round(final_score*100,0))
        final_scores.append(new)
        i+=1

    return final_scores

def compute_recommendations_csv():
    start_time = time.time()
    df = pd.read_csv("movie_dataset.csv")
    all_movie_titles = df["title"].to_numpy() # Array with the name of all movies from the dataframe

    features = ['keywords', 'cast', 'genres', 'director']

    for feature in features:
        df[feature] = df[feature].fillna("")

    df["combined_features"] = df.apply(combine_features, axis=1)

    cv = CountVectorizer()

    count_matrix_keywords = cv.fit_transform(df["keywords"])
    count_matrix_cast = cv.fit_transform(df["cast"])
    count_matrix_genres = cv.fit_transform(df["genres"])
    count_matrix_director = cv.fit_transform(df["director"])

    # Compute the Cosine Similatiry based on the count_matrix
    cosine_sim_keywords = cosine_similarity(count_matrix_keywords)
    cosine_sim_cast = cosine_similarity(count_matrix_cast)
    cosine_sim_genres = cosine_similarity(count_matrix_genres)
    cosine_sim_director = cosine_similarity(count_matrix_director)


    # open the file in the write mode
    with open('recommendations_values.csv', 'w', newline='') as file:
        # create the csv writer
        writer = csv.writer(file)

        header = []
        header.append("movies")
        header.extend(all_movie_titles)

        # write the header to the csv file
        writer.writerow(header)

        i = 0
        for movie in all_movie_titles:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Computing recommendations...")
            print("Computed movies: ", i)
            print("Progress: ", round((i * 100) / len(all_movie_titles),2) , "%")
            print("Analyzing: ", movie)

            recommendation_values = get_recommendation(movie, cosine_sim_keywords, cosine_sim_cast, cosine_sim_genres, cosine_sim_director)
            new_row = []
            new_row.append(movie)

            for values in recommendation_values:
                new_row.append(values[1])

            writer.writerow(new_row)
            i+=1

    print("--- Recommendations computed in %s seconds ---" % (time.time() - start_time))

# This function computes the cosine of the features we care about
def compute_constant_values():
    df = pd.read_csv("movie_dataset.csv")

    all_movie_titles = df["title"].to_numpy() # Array with the name of all movies from the dataframe # Type: <class 'numpy.ndarray'>
    
    features = ['keywords', 'cast', 'genres', 'director']

    for feature in features:
        df[feature] = df[feature].fillna("")

    df["combined_features"] = df.apply(combine_features, axis=1)

    cv = CountVectorizer()

    count_matrix_keywords = cv.fit_transform(df["keywords"])
    count_matrix_cast = cv.fit_transform(df["cast"])
    count_matrix_genres = cv.fit_transform(df["genres"])
    count_matrix_director = cv.fit_transform(df["director"])

    # Compute the Cosine Similatiry based on the count_matrix
    cosine_sim_keywords = cosine_similarity(count_matrix_keywords) # Type: <class 'numpy.ndarray'>
    cosine_sim_cast = cosine_similarity(count_matrix_cast) # Type: <class 'numpy.ndarray'>
    cosine_sim_genres = cosine_similarity(count_matrix_genres) # Type: <class 'numpy.ndarray'>
    cosine_sim_director = cosine_similarity(count_matrix_director) # Type: <class 'numpy.ndarray'>

    np_dataframe = df.to_numpy()

    # save constants to npy file
    save('np_dataframe.npy', np_dataframe)
    save('all_movie_titles.npy', all_movie_titles)
    save('cosine_sim_keywords.npy', cosine_sim_keywords)
    save('cosine_sim_cast.npy', cosine_sim_cast)
    save('cosine_sim_genres.npy', cosine_sim_genres)
    save('cosine_sim_director.npy', cosine_sim_director)

'''
This function generates a dictionary stored on a npy file containing all the recomendations(>=50) for each movie.
'''
def generate_recommendations_dict_file():
    start_time = time.time()
    df = pd.read_csv("movie_dataset.csv")
    all_movie_titles = df["title"].to_numpy() # Array with the name of all movies from the dataframe

    recommendations_dict = {}

    i = 0
    for movie in all_movie_titles:
        i+=1
        movie_recommendations = get_recommendation_server(movie)
        recommendations_dict[str(movie)] = movie_recommendations

        os.system('cls' if os.name == 'nt' else 'clear') # Clear the terminal
        print("Execution time:", round((time.time() - start_time), 2), "s")
        print("Computed movies: ", i)
        print("Computed percentage: ", round((100*i)/len(all_movie_titles),2),"%" )

    # Generate npy file containing the dictionary with the computed recommendations
    np.save("recommendations_dictionary.npy", recommendations_dict)

    print("recommendations_dictionary.npy file generated successfully")
    print("--- Recommendations computed in %s seconds ---" % (time.time() - start_time))

def testin():
    """
    Use this when we want to update the values from the calculated cosines
    compute_constant_values()
    """

    # Get recommendations for a single movie
    # a = get_recommendation_server("Focus")
    # print(a)

    # Get recommendations for many movies: This is not a very efficient way of doing it
    # liked_movies = {"Inception", "The Game", "Focus", "The Truman Show", "The Wolf of Wall Street", "Zodiac", "The Social Network", "Dead Poets Society", "Fight Club", "The Blind Side"}
    # print(get_recommendation_from_list_server(liked_movies))

    # Compute the recommendations for all movies -> Create a dic structure -> Store the dic on a npy file
    # generate_recommendations_dict_file()

    liked_movies = {"Inception", "The Game", "Focus", "The Truman Show", "The Wolf of Wall Street", "Zodiac", "The Social Network", "Dead Poets Society", "Fight Club", "The Blind Side"}
    recons = get_recommendations_from_npy_file(liked_movies)

    i=0
    for x in recons:
        print(x, "->", recons[x])
        i+=1
        if i == 10:
            break

