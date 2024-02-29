import pandas as pd
import numpy as np 
import streamlit as st
import requests
import pickle

similarity_spacy = pickle.load(open('spacy_matrix.pkl', 'rb'))
similarity_cv = pickle.load(open('cv_matrix.pkl', 'rb'))
movies = pickle.load(open('films_list.pkl', 'rb'))
movies_list = movies['title'].values
full_dataset = pd.read_csv('dataset.csv')


def get_poster(id):
    url = f"https://api.themoviedb.org/3/movie/{id}?api_key=6f3c8acc14a993f76635fcaea30623c8"
    res = requests.get(url)
    data = res.json()
    partial_poster_path = data['poster_path']
    full_poster_path = "https://image.tmdb.org/t/p/original/" + partial_poster_path
    return full_poster_path

def get_recommendations(film, model):
    if model == 'spaCy Large Model':
        similarity = similarity_spacy
    else:
        similarity = similarity_cv
    index = movies[movies['title'] == film].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda sim:sim[1])
    recommended_films = []
    recommended_posters = []
    recommended_descriptions = []
    for i, _ in distance[1:6]:
        movie_id = movies.iloc[i].id 
        recommended_films.append(movies.iloc[i].title)
        recommended_posters.append(get_poster(movie_id))
        recommended_descriptions.append(full_dataset.iloc[i].overview)
    return recommended_films, recommended_posters, recommended_descriptions

header = st.container()
entry_box = st.container()
model_select = st.container()

with header:
    st.title("Film Recommendation System")

with entry_box:
    user_input = st.selectbox("Select film from dropdown", movies_list)

with model_select:
    model_choice = st.selectbox("Select model", ["Count Vectoriser", "spaCy Large Model"])

if st.button("Find Recommendations"):
    film_names, film_poster, film_descriptions = get_recommendations(user_input, model_choice)
    col_one, col_two, col_three, col_four, col_five = st.columns(5)
    with col_one:
        st.text(film_names[0])
        st.image(film_poster[0])
        st.text_area("Description", film_descriptions[0], height=300)
    with col_two:
        st.text(film_names[1])
        st.image(film_poster[1])
        st.text_area("Description", film_descriptions[1], height=300)
    with col_three:
        st.text(film_names[2])
        st.image(film_poster[2])
        st.text_area("Description", film_descriptions[2], height=300)
    with col_four:
        st.text(film_names[3])
        st.image(film_poster[3])
        st.text_area("Description", film_descriptions[3], height=300)
    with col_five:
        st.text(film_names[4])
        st.image(film_poster[4])
        st.text_area("Description", film_descriptions[4], height=300)
