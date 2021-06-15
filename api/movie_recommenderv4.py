'''
This version of the recommender returns a csv with all the values computed
for all the movies, so the consulting time drops to constant.
* This version generates a csv with all computed recommendations
* This script was highly focused on improving execution time
the first time was executed took 120 minutes, after performace improvments takes 2 minutes :)
'''

import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

def get_recommendation_server(liked_movie, sorted):
    df = pd.read_csv("movie_dataset.csv")
    all_movie_titles = df["title"].to_numpy() # Array with the name of all movies from the dataframe

    #Para hacer mas rapidas las consultas: Encontrar una manera de hacer que no se calucle todo esto del cosine y se acceda como constante
    # ya que mientras tengamos el mismo dataset, siempre va a ser lo mismo
    
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

    try:
        movie_index = get_index_from_title(liked_movie, df) #Gustos del usuario
    except:
        # Return false if movie was not found
        return False
    
    similar_movies_keywords = list(enumerate(cosine_sim_keywords[movie_index]))
    similar_movies_cast = list(enumerate(cosine_sim_cast[movie_index]))
    similar_movies_genres = list(enumerate(cosine_sim_genres[movie_index]))
    similar_movies_director = list(enumerate(cosine_sim_director[movie_index]))

    final_scores = get_final_scores_server(all_movie_titles, similar_movies_keywords, similar_movies_cast, similar_movies_genres, similar_movies_director)

    if sorted == False:
        return final_scores
      
    sorted_final_scores = sorted(final_scores, key=lambda x:x[1], reverse= True) # Peliculas ordernadas de mayor a menor gusto
    # print("--- Recommendations computed in %s seconds ---" % (time.time() - start_time))
    return sorted_final_scores

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

# Not finished...
def get_recommendation_from_CSV(movie_title):
    # En pausa hasta encontrar una manera mas eficiente de hacer las consultas
    start_time = time.time()
    data = pd.read_csv('recommendations_values.csv')
    header = data.columns

    print(header[0])
    print(header[1])

    print("--- result retrieved in %s seconds ---" % (time.time() - start_time))

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