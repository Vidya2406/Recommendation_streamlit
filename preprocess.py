# preprocess.py

import pandas as pd
import ast
import os

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "data"

MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")
CREDITS_PATH = os.path.join(DATA_DIR, "credits.csv")
KEYWORDS_PATH = os.path.join(DATA_DIR, "keywords.csv")

OUTPUT_PATH = os.path.join(DATA_DIR, "movies_cleaned.csv")


# -----------------------------
# Helper Functions
# -----------------------------

def convert(text):
    """
    Convert JSON-like string to list of names.
    """
    result = []

    try:
        data = ast.literal_eval(text)

        for item in data:
            result.append(item["name"])

    except:
        return []

    return result


def fetch_director(text):
    """
    Extract director from crew list.
    """
    result = []

    try:
        data = ast.literal_eval(text)

        for item in data:
            if item["job"] == "Director":
                result.append(item["name"])
                break

    except:
        return []

    return result


def fetch_cast(text):
    """
    Extract first 3 cast members.
    """
    result = []

    try:
        data = ast.literal_eval(text)

        count = 0

        for item in data:
            result.append(item["name"])

            count += 1

            if count == 3:
                break

    except:
        return []

    return result


def remove_spaces(items):
    """
    Remove spaces from words.
    Example:
    Robert Downey Jr
    ->
    RobertDowneyJr
    """

    return [i.replace(" ", "") for i in items]


# -----------------------------
# Load Data
# -----------------------------

print("Loading datasets...")

movies = pd.read_csv(MOVIES_PATH)
credits = pd.read_csv(CREDITS_PATH)
keywords = pd.read_csv(KEYWORDS_PATH)

print("Datasets loaded successfully.")


# -----------------------------
# Merge Data
# -----------------------------

movies = movies.merge(credits, on="title")
movies = movies.merge(keywords, on="id")


# -----------------------------
# Select Useful Columns
# -----------------------------

movies = movies[
    [
        "id",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew",
        "vote_average",
        "release_date",
        "runtime",
    ]
]


# -----------------------------
# Handle Missing Values
# -----------------------------

movies.dropna(inplace=True)

print("Missing values removed.")


# -----------------------------
# Feature Extraction
# -----------------------------

movies["genres"] = movies["genres"].apply(convert)

movies["keywords"] = movies["keywords"].apply(convert)

movies["cast"] = movies["cast"].apply(fetch_cast)

movies["crew"] = movies["crew"].apply(fetch_director)

movies["overview"] = movies["overview"].apply(lambda x: x.split())


# Remove Spaces

movies["genres"] = movies["genres"].apply(remove_spaces)

movies["keywords"] = movies["keywords"].apply(remove_spaces)

movies["cast"] = movies["cast"].apply(remove_spaces)

movies["crew"] = movies["crew"].apply(remove_spaces)


# -----------------------------
# Create Tags Column
# -----------------------------

movies["tags"] = (
    movies["overview"]
    + movies["genres"]
    + movies["keywords"]
    + movies["cast"]
    + movies["crew"]
)

# Join words

movies["tags"] = movies["tags"].apply(lambda x: " ".join(x))

# Lowercase

movies["tags"] = movies["tags"].apply(lambda x: x.lower())


# -----------------------------
# Release Year
# -----------------------------

movies["release_year"] = pd.to_datetime(
    movies["release_date"],
    errors="coerce"
).dt.year


# -----------------------------
# Final Dataset
# -----------------------------

final_df = movies[
    [
        "id",
        "title",
        "tags",
        "vote_average",
        "release_year",
        "runtime",
    ]
]

# Rename

final_df.rename(
    columns={
        "vote_average": "rating"
    },
    inplace=True
)

# Remove duplicates

final_df.drop_duplicates(
    subset="title",
    inplace=True
)

# Reset index

final_df.reset_index(
    drop=True,
    inplace=True
)


# -----------------------------
# Save Dataset
# -----------------------------

final_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n----------------------------------")
print("Preprocessing Completed Successfully")
print("----------------------------------")

print(f"Movies Processed : {len(final_df)}")
print(f"Saved File       : {OUTPUT_PATH}")

print("\nSample Data:\n")

print(final_df.head())
