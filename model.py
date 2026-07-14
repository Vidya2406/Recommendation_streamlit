# model.py

import os
import pickle
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATA_PATH = "data/movies_cleaned.csv"
MODEL_DIR = "models"

MOVIE_LIST_FILE = os.path.join(MODEL_DIR, "movie_list.pkl")
SIMILARITY_FILE = os.path.join(MODEL_DIR, "similarity.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.pkl")

# -------------------------------------------------
# Create models directory
# -------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------
# Load dataset
# -------------------------------------------------

print("Loading cleaned dataset...")

movies = pd.read_csv(DATA_PATH)

print(f"Dataset Loaded Successfully.")
print(f"Total Movies : {len(movies)}")

# -------------------------------------------------
# Replace missing tags
# -------------------------------------------------

movies["tags"] = movies["tags"].fillna("")

# -------------------------------------------------
# Vectorization
# -------------------------------------------------

print("\nCreating CountVectorizer...")

vectorizer = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

vectors = vectorizer.fit_transform(movies["tags"]).toarray()

print("Vectorization Complete.")

# -------------------------------------------------
# Cosine Similarity
# -------------------------------------------------

print("Calculating Similarity Matrix...")

similarity = cosine_similarity(vectors)

print("Similarity Matrix Created.")

# -------------------------------------------------
# Save Files
# -------------------------------------------------

print("\nSaving model files...")

with open(MOVIE_LIST_FILE, "wb") as f:
    pickle.dump(movies, f)

with open(SIMILARITY_FILE, "wb") as f:
    pickle.dump(similarity, f)

with open(VECTORIZER_FILE, "wb") as f:
    pickle.dump(vectorizer, f)

print("All files saved successfully.")

# -------------------------------------------------
# Test Recommendation
# -------------------------------------------------

def recommend(movie_name):
    """
    Returns top 5 recommended movies.
    """

    movie_name = movie_name.lower()

    movie_indices = movies[
        movies["title"].str.lower() == movie_name
    ].index

    if len(movie_indices) == 0:
        print("\nMovie not found.")
        return

    index = movie_indices[0]

    distances = list(enumerate(similarity[index]))

    distances = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    print("\nRecommended Movies:\n")

    for i, (_, score) in enumerate(distances, start=1):
        title = movies.iloc[_]["title"]
        print(f"{i}. {title} ({score:.3f})")

# -------------------------------------------------
# Example
# -------------------------------------------------

if __name__ == "__main__":

    print("\n-----------------------------------")
    print("Movie Recommendation Model Ready")
    print("-----------------------------------")

    while True:

        movie = input("\nEnter movie name (or 'exit'): ")

        if movie.lower() == "exit":
            break

        recommend(movie)

    print("\nGoodbye!")
