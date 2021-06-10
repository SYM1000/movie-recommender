'''
En esta version del recomendador ha cambiado:
1. Separamos las features de cada pelicula en categorias diferentes que se aagregan a la tabla
se busca hacer que funcione con varias peliculas(Gustos del usuario)
'''

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###### Helper Function. Use them when needed ######
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File
df = pd.read_csv("movie_dataset.csv")
#print(df.columns)

##Step 2: Select Features

features = ['keywords', 'cast', 'genres', 'director']
##Step 3: Create a column in DF wich combines all selected features
for feature in features:
    df[feature] = df[feature].fillna("") #Loop over features an dfill all Nan


def combine_features(row):
    try:
        return row['keywords'] +" "+ row['cast']+" "+row["genres"]+" "+row["director"]
    except:
        print("Error", row)

df["combined_features"] = df.apply(combine_features, axis=1)
#print("Combined Features:" , df["combined_features"].head())
##Step 4: Create count matrix from this new combined column
cv = CountVectorizer()

count_matrix_keywords = cv.fit_transform(df["keywords"])
count_matrix_cast = cv.fit_transform(df["cast"])
count_matrix_genres = cv.fit_transform(df["genres"])
count_matrix_director = cv.fit_transform(df["director"])

##Step 5: Compute the Cosine Similatiry based on the count_matrix
cosine_sim_keywords = cosine_similarity(count_matrix_keywords)
cosine_sim_cast = cosine_similarity(count_matrix_cast)
cosine_sim_genres = cosine_similarity(count_matrix_genres)
cosine_sim_director = cosine_similarity(count_matrix_director)

movie_user_likes = "Inglorious Basterds"

##Step 6: Get index of this movie from its title
try:
    movie_index = get_index_from_title(movie_user_likes) #Gustos del usuario
except:
    print("Pelicula no encontrada\n")
    exit()

similar_movies_keywords = list(enumerate(cosine_sim_keywords[movie_index]))
similar_movies_cast = list(enumerate(cosine_sim_cast[movie_index]))
similar_movies_genres = list(enumerate(cosine_sim_genres[movie_index]))
similar_movies_director = list(enumerate(cosine_sim_director[movie_index]))

##Step 7: Get a list of similar movies in descending order of similarity scores
sorted_similar_movies_keywords = sorted(similar_movies_keywords, key=lambda x:x[1], reverse= True)
sorted_similar_movies_cast = sorted(similar_movies_cast, key=lambda x:x[1], reverse= True)
sorted_similar_movies_genres = sorted(similar_movies_genres, key=lambda x:x[1], reverse= True)
sorted_similar_movies_director = sorted(similar_movies_director, key=lambda x:x[1], reverse= True)


#Step 8: Get scores per feature for each movie and adding each feature_score to the csv
def get_array_of_values(similar_movies2):
    new_array = []

    for x in similar_movies2:
        v = x[1]
        new_array.append(v)

    return new_array

keywords_scores = sorted_similar_movies_keywords
cast_scores = sorted_similar_movies_cast
genres_scores = sorted_similar_movies_genres
director_scores = sorted_similar_movies_director

final_scores = []

def get_aray_of_tuples_with_final_scores():
    i = 0
    for movie in similar_movies_keywords:

        index = movie[0]
        #otro = similar_movies_director[0][0]

        keyword_score = similar_movies_keywords[i][1]
        cast_score = similar_movies_cast[i][1]
        genres_score = similar_movies_genres[i][1]
        director_score = similar_movies_director[i][1]

        final_score = ( (keyword_score * 0.00) + (cast_score * 0.20) + (genres_score * 0.60) + (director_score * 0.20) )
        new = (index, final_score)
        final_scores.append(new)
        i+=1

get_aray_of_tuples_with_final_scores()
sorted_final_scores = sorted(final_scores, key=lambda x:x[1], reverse= True)
#print(sorted_final_scores)

'''
keywords_scores = get_array_of_values(similar_movies_keywords)
cast_scores = get_array_of_values(similar_movies_cast)
genres_scores = get_array_of_values(similar_movies_genres)
director_scores = get_array_of_values(similar_movies_director)


try:
    df["keywords_scores"] = keywords_scores
    df["cast_scores"] = cast_scores
    df["genres_scores"] = genres_scores
    df["director_scores"] = director_scores

except:
    print("error al agregar scores al dataframe")
'''

#Aqui creamos una columna con el valor final de cada pelicula, tomando en cuenta el valor de cada feature (promedio en la primera version)
def get_final_score(row):
    try:
        keyword_score = row['keywords_scores']
        cast_score = row['cast_scores']
        genres_score = row['genres_scores']
        director_score = row['director_scores']

        return (keyword_score + cast_score + genres_score + director_score) / 4
    except:
        print("Error obteniendo score final", row)

#df["combined_score"] = df.apply(get_final_score, axis=1)

'''
print("Scores combinados")
print(df["combined_score"])
'''


##Step : Print titles of first 50 movies

print("Recomendaciones para la pelicula ", movie_user_likes)

'''
print("\nSimilar Keywords")
i=0
for movie in sorted_similar_movies_keywords:
    v = movie[1]*100
    print ( get_title_from_index(movie[0]), " ","%.2f" % v, "%")
    i= i+1
    if i>5:
        break

print("\nSimilar cast")
i=0
for movie in sorted_similar_movies_cast:
    v = movie[1]*100
    print ( get_title_from_index(movie[0]), " ","%.2f" % v, "%")
    i= i+1
    if i>5:
        break


print("\nSimilar Genres")
i=0
for movie in sorted_similar_movies_genres:
    v = movie[1]*100
    print ( get_title_from_index(movie[0]), " ","%.2f" % v, "%")
    i= i+1
    if i>5:
        break


print("\nSimilar Director")
i=0
for movie in sorted_similar_movies_director:
    v = movie[1]*100
    print ( get_title_from_index(movie[0]), " ","%.2f" % v, "%")
    i= i+1
    if i>5:
        break
'''




print("\nFinal Scores")
i=0
for movie in sorted_final_scores:
    v = movie[1]*100
    print ( get_title_from_index(movie[0]), " ","%.2f" % v, "%")
    i= i+1
    if i>10:
        break
