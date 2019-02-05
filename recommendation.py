import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Recommendation():
    dataframe = None
    similarity = None
    
    def __init__(self):
        self.dataframe = pd.read_csv("./movie_metadata.csv",sep=',',low_memory=False).drop_duplicates(subset='movie_title')        
        
        self.dataframe['title_year'].fillna('', inplace=True)
        self.dataframe['movie_title'] = self.dataframe.apply(self.edit_name, axis=1)
        self.dataframe['director_name'] = self.dataframe.apply(self.edit_director, axis=1)
        self.dataframe['plot_keywords'] = self.dataframe.apply(self.edit_keywords, axis=1)
        self.dataframe['genres'] = self.dataframe.apply(self.edit_genres, axis=1)
        self.dataframe['title_year'] = self.dataframe.apply(self.edit_title_year, axis=1)
        self.dataframe['soup'] = self.dataframe.apply(self.create_soup, axis=1)
        self.dataframe.drop(self.dataframe[(self.dataframe['director_name'] == '') & (self.dataframe['title_year'] == '')].index,inplace=True)
    
        count_vectorizer = CountVectorizer(stop_words='english')
        count_vectorizer_matrix = count_vectorizer.fit_transform(self.dataframe['soup'])
        self.similarity = cosine_similarity(count_vectorizer_matrix, count_vectorizer_matrix)

    def recommendation_movies(self,name_movie):
        try:
            indices = pd.Series(self.dataframe.index, index=self.dataframe['movie_title'])
            movies = indices[name_movie]
            similarity_scores = list(enumerate(self.similarity[movies]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            similarity_scores = similarity_scores[1:20]
            indices_movies = [i[0] for i in similarity_scores]
            print(self.dataframe[['movie_title','director_name','duration','title_year']].iloc[indices_movies])
        except Exception:
            movie_suggestion = self.similar_name(name_movie)
            print("Movie not Found")
            print("Movie Suggestion:")
            for movie in movie_suggestion:
                print(movie)   

    def similar_name(self,name_movie):
        movies = self.dataframe
        movies['similar'] = movies['movie_title'].apply(lambda movie: SequenceMatcher(None,name_movie,movie).ratio())
        return movies[movies['similar'] > 0.7]['movie_title'].values    

    def edit_name(self,movie):
        name = movie['movie_title'].strip(" ").replace('\xa0', '')
        return name

    def edit_director(self,movie):
        if isinstance(movie['director_name'], str):
            return movie['director_name']
        else:
            return ''
    
    def edit_keywords(self,movie):
        if isinstance(movie['plot_keywords'], str):
            keywords = movie['plot_keywords'].split("|")
            return keywords
        else:
            return ''
    
    def edit_genres(self,movie):
        if isinstance(movie['genres'], str):
            genres = movie['genres'].split("|")
            return genres
        else:
            return ''

    def edit_title_year(self, movie):
        return str(movie['title_year'])   

    def create_soup(self,movie):        
        return  ''.join(movie['movie_title']) + ' ' + ' '.join(movie['director_name']) + ' ' + ' '.join(movie['plot_keywords']) + ' ' + ' '.join(movie['genres']) + ' ' + ' '.join(movie['title_year'])
