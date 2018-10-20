import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(title,cosine):
    try:
        indices = pd.Series(metadata.index, index=metadata['movie_title'])

        idx = indices[title]

        sim_scores = list(enumerate(cosine[idx]))

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        sim_scores = sim_scores[1:20]

        movie_indices = [i[0] for i in sim_scores]

        return metadata['movie_title'].iloc[movie_indices]
    except Exception as e:
        return "Movie not found"
   
def edit_name(movie):
    new_name = movie['movie_title'].strip(" ").replace('\xa0','')
    return new_name  

def edit_director(movie):
    if isinstance(movie['director_name'],str):
        return movie['director_name']
    else:
        return ''    

def edit_genres(movie):
    if isinstance(movie['genres'],str):
        genres = movie['genres'].split("|")
        return genres
    else:
        return ''    

def edit_keywords(movie):
    if isinstance(movie['plot_keywords'],str):
        keywords = movie['plot_keywords'].split("|")
        return keywords
    else:
        return ''

def create_soup(movie):
    return ' '.join(movie['movie_title']) + ' ' + ' '.join(movie['director_name']) + ' ' + ' '.join(movie['plot_keywords']) + ' ' + ' '.join(movie['genres'])        

if len(sys.argv) >= 2:
    name = ''

    if len(sys.argv) == 2:
        name = sys.argv[1]
    else:
        del(sys.argv[0])
        name = ' '.join(sys.argv)    

    metadata = pd.read_csv("./movie_metadata.csv",sep = ',',low_memory=False).drop_duplicates(subset='movie_title')

    tfidf = CountVectorizer()

    metadata['movie_title'] = metadata.apply(edit_name,axis=1)

    metadata['director_name'] = metadata.apply(edit_director,axis=1)

    metadata['genres'] = metadata.apply(edit_genres,axis=1)

    metadata['plot_keywords'] = metadata.apply(edit_keywords,axis=1)

    metadata['soup'] = metadata.apply(create_soup,axis=1)

    tfidf_matrix = tfidf.fit_transform(metadata['soup'])

    cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)

    movies = get_recommendations(name,cosine_sim)    

    print (movies)

else:
    print ("Args with movie name is required")

