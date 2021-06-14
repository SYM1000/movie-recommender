'''
En esta version del recomendador vamos darle soporte 
para hacer el llamado y que retorne todas las peliculas 
recomendadas con una sola funcion y de esa manera poderla 
llamar desde el script 'app_api'
'''

'''
IDEA: Una forma para mejorar el tiempo de ejecicion de las 
recomendaciones; dando como resultado en respuestas mas rapidas es:
crear un csv con las peliculas recomendadas para cada pelicula, en 
otras palabras computar todas las recomendaciones y guardar los
resultados en un CSV para devolver los resultados en tiempo constante.
'''

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Main function to call when we want to get the recommende movies for
# a single liked movie

def get_recommendation(liked_movie):
    df = pd.read_csv("movie_dataset.csv")

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

    sorted_similar_movies_keywords = sorted(similar_movies_keywords, key=lambda x:x[1], reverse= True)
    sorted_similar_movies_cast = sorted(similar_movies_cast, key=lambda x:x[1], reverse= True)
    sorted_similar_movies_genres = sorted(similar_movies_genres, key=lambda x:x[1], reverse= True)
    sorted_similar_movies_director = sorted(similar_movies_director, key=lambda x:x[1], reverse= True)

    keywords_scores = sorted_similar_movies_keywords
    cast_scores = sorted_similar_movies_cast
    genres_scores = sorted_similar_movies_genres
    director_scores = sorted_similar_movies_director

    final_scores = []

    get_aray_of_tuples_with_final_scores(similar_movies_keywords, similar_movies_cast, similar_movies_genres, similar_movies_director, final_scores) 

    sorted_final_scores = sorted(final_scores, key=lambda x:x[1], reverse= True)

    number_of_recommendations = 10 # this is the maximum number of recommendations we want

    movie_recommendations = [] # This is the final result

    i=0
    for movie in sorted_final_scores:
        # Check if movie has been seen
        if get_title_from_index(movie[0],df) == liked_movie:
            continue

        recommended_movie = []
        v = movie[1]*100
        # print ( get_title_from_index(movie[0],df), " ","%.2f" % v, "%")
        recommended_movie.append(get_title_from_index(movie[0],df))
        recommended_movie.append(round(v,2))

        movie_recommendations.append(recommended_movie)

        i= i+1
        if i>number_of_recommendations:
            break

    return movie_recommendations


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

def get_aray_of_tuples_with_final_scores(similar_movies_keywords, similar_movies_cast, similar_movies_genres, similar_movies_director, final_scores):
    i = 0
    for movie in similar_movies_keywords:

        index = movie[0]

        keyword_score = similar_movies_keywords[i][1]
        cast_score = similar_movies_cast[i][1]
        genres_score = similar_movies_genres[i][1]
        director_score = similar_movies_director[i][1]

        final_score = ( (keyword_score * 0.00) + (cast_score * 0.20) + (genres_score * 0.60) + (director_score * 0.20) )
        new = (index, final_score)
        final_scores.append(new)
        i+=1

# Testing
start_time = time.time()
for _ in range(10):
    get_recommendation("Avatar")

print("--- %s seconds ---" % (time.time() - start_time))
# print(get_recommendation("Avatar"))