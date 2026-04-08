import streamlit as st
import pickle 
import pandas as pd

data = pickle.load(open('data.pkl', mode = 'rb'))
data = pd.DataFrame(data)

similarity = pickle.load(open('similarity.pkl', mode = 'rb'))

def recommend(movie):
    movie_row = data[data['title'] == movie.lower().replace(" ", "")]

    if movie_row.empty:
        st.warning("The movie you are requesting is not present")
        return
    else:
        movie_index = movie_row.index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse = True, key = lambda x: x[1] )[1:6]
  
    return movie_list


st.set_page_config(page_title="MRS(Content Based Filtering)", layout='wide')

st.title(":film_projector: :blue[Movie Recommendation System]")
st.sidebar.title(":red[Project Overview]")
st.sidebar.write("A content-based movie recommendation system that suggests movies similar \
                 to your selection using genres, keywords, and overview. ")
st.sidebar.write("It uses TF-IDF vectorization and cosine similarity to find and display the top 5 most relevant movies.")

selected_movie = st.selectbox("Choose your movie: ", data['original_title'].values)

button = st.button(":red[recommend]")

if button:
    movie_list = recommend(selected_movie)
    for i in movie_list:
        st.write(data.iloc[i[0]].original_title)