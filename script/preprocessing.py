import ast
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

en_stop = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_genres(dct):
    return [i['name'] for i in dct]

def clean_keywords(dct):
    return [i['name'] for i in dct]

def clean_cast(dct):
    # Only keep the first 3 cast members
    return [i['name'] for idx, i in enumerate(dct) if idx < 3]

def clean_crew(dct):
    # Keep only the Director
    return [i['name'] for i in dct if i['job'] == 'Director']

def handle_data(lst):
    # Remove spaces and lower text
    return [i.lower().replace(" ", "") for i in lst]

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

def remove_stopwords(lst):
    return [word for word in lst if word not in en_stop]

def apply_stemming(lst):
    return [ps.stem(word) for word in lst]

def preprocess_data(data):
    """
    Cleans structural string fields using AST, extracts relevant nested data,
    combines text fields, cleans up language (punctuation, stopwords, stemming),
    and reduces down to target tags.
    """
    print("Preprocessing data items (genres, keywords, cast, crew)...")
    data = data.copy()

    # Filter out empty entries first
    for i in data.select_dtypes(include='object').columns:
        data = data[data[i] != '[]']

    # Convert stringified JSON to proper dictionaries
    lst_cols = ['genres', 'keywords', 'cast', 'crew']
    for col in lst_cols:
        data[col] = data[col].apply(ast.literal_eval)

    # Extract names
    data['genres'] = data['genres'].apply(clean_genres)
    data['keywords'] = data['keywords'].apply(clean_keywords)
    data['cast'] = data['cast'].apply(clean_cast)
    data['crew'] = data['crew'].apply(clean_crew)

    # Normalize (lowercase, remove spaces)
    data['cast'] = data['cast'].apply(handle_data)
    data['crew'] = data['crew'].apply(handle_data)
    data['genres'] = data['genres'].apply(handle_data)
    data['keywords'] = data['keywords'].apply(handle_data)

    # Keep original format of title before modification
    data['original_title'] = data['title']
    data['title'] = data['title'].str.lower().str.replace(" ", "")

    print("Building intermediate tags combining components...")
    # Weigh components (Genres * 4, Crew * 3, Keywords * 2)
    data['temp'] = data['title'].str.split(" ") + data['genres']*4 + data['keywords']*2 + data['cast'] + data['crew']*3

    print("Cleaning overviews...")
    # Process overview text
    data['overview'] = data['overview'].str.lower().str.split()
    data['overview'] = data['overview'].apply(remove_punc)
    data['overview'] = data['overview'].apply(remove_stopwords)
    data['overview'] = data['overview'].apply(apply_stemming)

    # Combine into unified feature
    data['tags'] = data['temp'] + data['overview']

    print("Dropping legacy columns and duplicates...")
    # Drop independent columns as we now have tags
    data.drop(columns=['genres', 'overview', 'keywords', 'cast', 'crew', 'temp'], inplace=True)
    
    # Merge list into strings
    data['tags'] = data['tags'].apply(lambda x: " ".join(x))

    # Keep distinct entries and rest index
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data
