# utils.py
import os
import pickle
import requests
import streamlit as st

# -------------------------------------------------------
# Path setup — anchored to this file's location, not cwd
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MOVIE_MODEL = os.path.join(MODEL_DIR, "movie_list.pkl")
SIMILARITY_MODEL = os.path.join(MODEL_DIR, "similarity.pkl")

# If your files are too large for git, host them somewhere
# (Google Drive, Hugging Face, S3) and set direct download URLs here.
# Leave blank ("") if your files are committed to the repo directly.
MOVIE_MODEL_URL = ""       # e.g. "https://huggingface.co/.../movie_list.pkl"
SIMILARITY_MODEL_URL = ""  # e.g. "https://huggingface.co/.../similarity.pkl"


def _download_if_missing(path, url):
    """Download a model file if it's not present locally."""
    if os.path.exists(path):
        return
    if not url:
        raise FileNotFoundError(
            f"Model file not found at {path} and no download URL was provided. "
            f"Make sure the file is committed to the repo, or set a download URL."
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with st.spinner(f"Downloading {os.path.basename(path)}..."):
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


# -------------------------------------------------------
# Cached model loading — loads once per session, not per rerun
# -------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        _download_if_missing(MOVIE_MODEL, MOVIE_MODEL_URL)
        _download_if_missing(SIMILARITY_MODEL, SIMILARITY_MODEL_URL)

        with open(MOVIE_MODEL, "rb") as f:
            movies = pickle.load(f)

        with open(SIMILARITY_MODEL, "rb") as f:
            similarity = pickle.load(f)

        return movies, similarity

    except FileNotFoundError as e:
        st.error(
            "Model files could not be loaded. Please check that "
            "movie_list.pkl and similarity.pkl are present in the "
            "'models' folder, or that a valid download URL is configured."
        )
        raise e
    except Exception as e:
        st.error(f"Unexpected error loading models: {e}")
        raise e


# -------------------------------------------------------
# Movie titles list
# -------------------------------------------------------
@st.cache_data
def movie_titles():
    movies, _ = load_models()
    return movies["title"].values


# -------------------------------------------------------
# Recommendation logic
# -------------------------------------------------------
def recommend(movie_title, top_n=10):
    movies, similarity = load_models()

    matches = movies[movies["title"] == movie_title]
    if matches.empty:
        return []

    index = matches.index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []
    for i, _score in distances[1: top_n + 1]:
        row = movies.iloc[i]
        recommendations.append({
            "title": row.get("title", "Unknown"),
            "poster": row.get("poster", ""),  # fallback handled in app.py
            "rating": row.get("rating", "N/A"),
            "year": row.get("year", "N/A"),
            "runtime": row.get("runtime", "N/A"),
        })

    return recommendations


# -------------------------------------------------------
# Sidebar statistics
# -------------------------------------------------------
@st.cache_data
def statistics():
    movies, _ = load_models()
    return {
        "movies": len(movies),
        "highest_rating": movies["rating"].max() if "rating" in movies else "N/A",
        "average_rating": round(movies["rating"].mean(), 2) if "rating" in movies else "N/A",
    }
