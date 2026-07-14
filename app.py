# app.py

import streamlit as st
from utils import (
    movie_titles,
    recommend,
    statistics
)

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------------------------------
# Custom CSS
# -------------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

h1{
    color:#FAFAFA;
    text-align:center;
}

.movie-card{
    background:#1E1E1E;
    padding:12px;
    border-radius:10px;
    text-align:center;
}

.movie-title{
    font-size:18px;
    font-weight:bold;
    color:white;
}

.movie-info{
    color:#BBBBBB;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.title("🎬 Movie Recommendation")

stats = statistics()

st.sidebar.metric(
    "Movies",
    stats["movies"]
)

st.sidebar.metric(
    "Highest Rating",
    stats["highest_rating"]
)

st.sidebar.metric(
    "Average Rating",
    stats["average_rating"]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
Content-Based Movie Recommendation System

Built using:

- Streamlit
- Scikit-Learn
- CountVectorizer
- Cosine Similarity
"""
)

# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.title("🎬 Movie Recommendation System")

st.write(
    "Select a movie to get similar movie recommendations."
)

# -------------------------------------------------------
# Movie Selection
# -------------------------------------------------------

movies = movie_titles()

selected_movie = st.selectbox(
    "Choose a Movie",
    movies
)

num_recommendations = st.slider(
    "Number of Recommendations",
    5,
    20,
    10
)

# -------------------------------------------------------
# Recommendation Button
# -------------------------------------------------------

if st.button("Recommend Movies"):

    with st.spinner("Finding similar movies..."):

        recommendations = recommend(
            selected_movie,
            top_n=num_recommendations
        )

    if len(recommendations) == 0:

        st.error("Movie not found.")

    else:

        st.success(
            f"Top {len(recommendations)} Recommendations"
        )

        cols = st.columns(5)

        for index, movie in enumerate(recommendations):

            with cols[index % 5]:

                st.image(
                    movie["poster"],
                    use_container_width=True
                )

                st.markdown(
                    f"""
<div class="movie-card">

<div class="movie-title">
{movie['title']}
</div>

<br>

<div class="movie-info">

⭐ Rating:
{movie['rating']}

</div>

<div class="movie-info">

📅 Year:
{movie['year']}

</div>

<div class="movie-info">

⏱ Runtime:
{movie['runtime']} min

</div>

</div>
""",
                    unsafe_allow_html=True
                )

# -------------------------------------------------------
# Footer
# -------------------------------------------------------

st.markdown("---")

st.markdown(
"""
<center>

Made with ❤️ using

Streamlit + Scikit-Learn

</center>
""",
unsafe_allow_html=True
)
