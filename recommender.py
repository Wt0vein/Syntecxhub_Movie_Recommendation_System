import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    movies = pd.read_csv('data/movies.csv')
    movies['tags'] = movies['overview'] + " " + movies['genres'] + " " + movies['keywords']
    return movies

def build_model(movies):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

def recommend(movie, movies, similarity):
    idx = movies[movies['title'] == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:4]
    return [movies.iloc[i[0]].title for i in movie_list]

if __name__ == "__main__":
    movies = load_data()
    similarity = build_model(movies)
    print(recommend("Avatar", movies, similarity))
