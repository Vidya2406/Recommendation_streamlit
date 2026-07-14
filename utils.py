# utils.py

import pickle
import requests
import streamlit as st

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

MOVIE_MODEL = "models/movie_list.pkl"
SIMILARITY_MODEL = "models/similarity.pkl"

# Replace with your TMDb API Key
TMDB_API_KEY = "YOUR_TMDB_API_KEY"

DEFAULT_POSTER = "https://via.placeholder.com/300x450?text=No+Poster"

# ----------------------------------------------------
# Load Models
# ----------------------------------------------------

@st.cache_resource
def load_models():
    """
    Load movie dataframe and similarity matrix.
    """

    with open(MOVIE_MODEL, "rb") as f:
        movies = pickle.load(f)

    with open(SIMILARITY_MODEL, "rb") as f:
        similarity = pickle.load(f)

    return movies, similarity


# ----------------------------------------------------
# Fetch Poster from TMDb
# ----------------------------------------------------

def fetch_poster(movie_name):
    """
    Fetch movie poster using TMDb Search API.
    """

    try:

        url = (
            "https://api.themoviedb.org/3/search/movie"
            f"?api_key={TMDB_API_KEY}"
            f"&query={movie_name}"
        )

        response = requests.get(url, timeout=10).json()

        results = response.get("results")

        if results:

            poster_path = results[0].get("poster_path")

            if poster_path:
                return (
                    "https://image.tmdb.org/t/p/w500"
                    + poster_path
                )

    except Exception:
        pass

    return DEFAULT_POSTER


# ----------------------------------------------------
# Fetch Movie Details
# ----------------------------------------------------

def fetch_movie_details(movie_name):
    """
    Get movie details from TMDb.
    """

    try:

        url = (
            "https://api.themoviedb.org/3/search/movie"
            f"?api_key={TMDB_API_KEY}"
            f"&query={movie_name}"
        )

        response = requests.get(url, timeout=10).json()

        results = response.get("results")

        if results:

            movie = results[0]

            return {

                "title": movie.get("title"),

                "rating": round(movie.get("vote_average", 0), 1),

                "overview": movie.get("overview", "No overview available."),

                "release": movie.get("release_date", "Unknown"),

                "poster": fetch_poster(movie_name)

            }

    except Exception:
        pass

    return {

        "title": movie_name,

        "rating": "N/A",

        "overview": "No description available.",

        "release": "Unknown",

        "poster": DEFAULT_POSTER

    }


# ----------------------------------------------------
# Recommend Movies
# ----------------------------------------------------

def recommend(movie_name, top_n=10):
    """
    Return top recommended movies.
    """

    movies, similarity = load_models()

    movie_name = movie_name.lower()

    indices = movies[
        movies["title"].str.lower() == movie_name
    ].index

    if len(indices) == 0:
        return []

    index = indices[0]

    distances = list(enumerate(similarity[index]))

    distances = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True
    )[1:top_n + 1]

    recommendations = []

    for i in distances:

        movie = movies.iloc[i[0]]

        recommendations.append({

            "title": movie["title"],

            "rating": movie.get("rating", "N/A"),

            "year": movie.get("release_year", ""),

            "runtime": movie.get("runtime", ""),

            "poster": fetch_poster(movie["title"])

        })

    return recommendations


# ----------------------------------------------------
# Movie List
# ----------------------------------------------------

def movie_titles():
    """
    Return sorted movie titles.
    """

    movies, _ = load_models()

    return sorted(movies["title"].tolist())


# ----------------------------------------------------
# Search Movies
# ----------------------------------------------------

def search_movies(keyword):
    """
    Search movie names.
    """

    movies, _ = load_models()

    keyword = keyword.lower()

    return sorted(
        movies[
            movies["title"].str.lower().str.contains(keyword)
        ]["title"].tolist()
    )


# ----------------------------------------------------
# Top Rated Movies
# ----------------------------------------------------

def top_rated(limit=10):
    """
    Return highest rated movies.
    """

    movies, _ = load_models()

    return (
        movies.sort_values(
            by="rating",
            ascending=False
        )
        .head(limit)
    )


# ----------------------------------------------------
# Statistics
# ----------------------------------------------------

def statistics():

    movies, _ = load_models()

    return {

        "movies": len(movies),

        "highest_rating": movies["rating"].max(),

        "average_rating": round(
            movies["rating"].mean(),
            2
        )

    }
