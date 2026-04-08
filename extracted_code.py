import numpy as np 
import pandas as pd

import pickle

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

credits = pd.read_csv('credits.csv')
credits_main = credits.copy()
credits

keywords = pd.read_csv('keywords.csv')
keywords_main = keywords.copy()
keywords

movies_metadata = pd.read_csv('movies_metadata.csv')
movies_metadata_main = movies_metadata.copy()
movies_metadata

movies_metadata.columns

drop_cols = ['adult', 'budget', 'popularity', 'production_countries','poster_path' ,'homepage', 'imdb_id', 'original_language', 'release_date','revenue','runtime','spoken_languages','status', 'original_title','video', 'vote_average','vote_count']

def remove_cols(drop_cols):
    lst = []
    for i in drop_cols:
        if i in movies_metadata.columns:
            lst.append(i)
        else:
            print(f"{i} is not present in the dataset")

    movies_metadata.drop(columns = lst, inplace = True)

remove_cols(drop_cols)

movies_metadata


movies_metadata['belongs_to_collection'].isnull().sum()
# since most of the rows of this column are null there is no particular use of it for our recommendation system
remove_cols(['belongs_to_collection'])

movies_metadata

for i in movies_metadata.columns:
    print(f'{i} column has {movies_metadata[i].isnull().sum()} null values')

remove_cols(['tagline'])
remove_cols(['production_companies'])

movies_metadata[movies_metadata['genres'] == '[]']

movies_metadata['id'] = pd.to_numeric(movies_metadata['id'], errors='coerce')
movies_metadata = movies_metadata.dropna(subset=['id'])

movies_metadata['id'] = movies_metadata['id'].astype('int64')

data = pd.merge(movies_metadata, keywords, on = 'id', how = 'inner')
data.head()

data = pd.merge(data,credits ,on = 'id', how = 'inner')
data.head()

data.shape

data.isnull().sum()

data.dropna(inplace = True)

data.isnull().sum()



data.info()

data[data['id']== 468343]

type(data['genres'][0])

data.select_dtypes(include = 'object').columns

for i in data.select_dtypes(include = 'object').columns:
    data = data[data[i] != '[]']





data.shape

lst_cols = ['genres','keywords', 'cast', 'crew' ]
import ast
for col in lst_cols:
    data[col] = data[col].apply(ast.literal_eval)

for i in lst_cols:
    print(type(data[i][862]))

def clean_genres(dct):
    lst = []
    for i in dct:
        lst.append(i['name'])
    return lst

data['genres'] = data['genres'].apply(clean_genres)



def clean_keywords(dct):
    lst = []
    for i in dct:
        lst.append(i['name'])
    return lst

data['keywords'] =  data['keywords'].apply(clean_keywords)

def clean_cast(dct):
    lst = []
    counter = 0
    
    for i in dct:
        if counter == 3:
            break
        lst.append(i['name'])
        counter += 1
    
    return lst

data['cast'] = data['cast'].apply(clean_cast)

def clean_crew(dct):
    lst = []
    for i in dct:
        if i['job'] == 'Director':
            lst.append(i['name'])
    return lst

data['crew'] = data['crew'].apply(clean_crew)

data.head()

type(data['overview'][0])

def handle_data(lst):
    l = []
    for i in lst:
        l.append(i.lower().replace(" ", ""))
    return l

data['cast'] = data['cast'].apply(handle_data)

data['crew'] = data['crew'].apply(handle_data)

data['genres'] = data['genres'].apply(handle_data)

data['keywords'] = data['keywords'].apply(handle_data)


data['original_title'] = data['title']

data['title'] = data['title'].str.lower().str.replace(" ", "")

data['temp'] = data['title'].str.split(" ") + data['genres']*4 + data['keywords']*2 + data['cast'] + data['crew']*3

data['temp'][0]

data['overview'] = data['overview'].str.lower().str.split()


data['overview'][0]

# # removing punctuation 
# import string

# def remove_punc(lst):
#     table = str.maketrans('', '', string.punctuation)
#     result = []
    
#     for text in lst:
#         cleaned = text.translate(table)
        
#         # choose ONLY one version
#         final_text = cleaned if cleaned != text else text
        
#         # avoid duplicates
#         if final_text not in result:
#             result.append(final_text)
    
#     return result

# remove_punc(data['overview'][0])

import string
def remove_punc(lst):
    l = []
    for word in lst:
        cleaned = word
        for char in string.punctuation:
            if char in word:
                cleaned = cleaned.replace(char, '')
        if cleaned not in l:
            l.append(cleaned)
    return l

remove_punc(data['overview'][0])
        

data['overview'] = data['overview'].apply(remove_punc)
data['overview']

len(data['overview'][0])

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

en_stop = stopwords.words('english')

def remove_stopwords(lst):
    l = []
    for word in lst:
        if word in en_stop:
            pass
        else:
            l.append(word)

    return l

remove_stopwords(data['overview'][0])

data['overview'] = data['overview'].apply(remove_stopwords)
data['overview']

len(data['overview'][0])

ps = PorterStemmer()
data['overview'] = data['overview'].apply(lambda words: [ps.stem(word) for word in words])



data['tags'] = data['temp'] + data['overview']

data['tags'][0]



data.columns

data.drop(columns = ['genres', 'overview', 'keywords', 'cast', 'crew', 'temp'], inplace = True)

data.head()

data['tags'] = data['tags'].apply(lambda x: " ".join(x))
data['tags']

data.isnull().sum()

data.duplicated().sum()

data.drop_duplicates(inplace = True)

data.reset_index(drop = True, inplace = True)

TfIdf = TfidfVectorizer(max_features = 5000)
TfIdf

vectors = TfIdf.fit_transform(data['tags']).toarray()

vectors

len(vectors[0])

similarity = cosine_similarity(vectors)

sorted(list(enumerate(similarity[0])),reverse = True, key = lambda x: x[1] )[1:6]

def recommend(movie):
    movie_row = data[data['title'] == movie.lower().replace(" ", "")]

    if movie_row.empty:
        print("The movie you are requesting is not present")
        return
    else:
        movie_index = movie_row.index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse = True, key = lambda x: x[1] )[1:6]

    for i in movie_list:
        print(data.iloc[i[0]].original_title)

recommend('the Dark Knight')

data['original_title'][:20]



import pickle
pickle.dump(data.to_dict(), open('data.pkl', mode = 'wb'))

pickle.dump(similarity, open('similarity.pkl', mode = 'wb'))

