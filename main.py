from recommendation import Recommendation

def menu():
    recommendation = Recommendation()
    name_movie = None
    while name_movie != "":
        print("#################################")
        print("RECOMMENDATION CONTENT BASED IMDB")
        name_movie = input("Enter name movie: ")
        if name_movie != "":
            recommendation.recommendation_movies(name_movie)

menu()