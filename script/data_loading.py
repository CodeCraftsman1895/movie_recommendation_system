import pandas as pd
from .config import CREDITS_CSV, KEYWORDS_CSV, MOVIES_METADATA_CSV

def remove_cols(df, drop_cols):
    """
    Safely removes multiple columns from the dataframe if they exist.
    """
    lst = [i for i in drop_cols if i in df.columns]
    df.drop(columns=lst, inplace=True)

def load_and_merge_data():
    """
    Loads credits, keywords, and movies metadata CSV files,
    performs initial memory reduction by dropping unused columns,
    and merges them into a single dataframe.
    """
    print("Loading datasets...")
    credits = pd.read_csv(CREDITS_CSV)
    keywords = pd.read_csv(KEYWORDS_CSV)
    movies_metadata = pd.read_csv(MOVIES_METADATA_CSV, low_memory=False)

    print("Dropping unused columns from movies_metadata...")
    drop_cols = [
        'adult', 'budget', 'popularity', 'production_countries', 'poster_path', 
        'homepage', 'imdb_id', 'original_language', 'release_date', 'revenue', 
        'runtime', 'spoken_languages', 'status', 'original_title', 'video', 
        'vote_average', 'vote_count', 'belongs_to_collection', 'tagline', 'production_companies'
    ]
    remove_cols(movies_metadata, drop_cols)

    print("Formatting IDs and resolving NA values...")
    # Convert 'id' column to numeric, dropping NaNs to facilitate merging
    movies_metadata['id'] = pd.to_numeric(movies_metadata['id'], errors='coerce')
    movies_metadata = movies_metadata.dropna(subset=['id'])
    movies_metadata['id'] = movies_metadata['id'].astype('int64')

    print("Merging datasets on ID...")
    data = pd.merge(movies_metadata, keywords, on='id', how='inner')
    data = pd.merge(data, credits, on='id', how='inner')
    
    # Drop rows that have NaN in the merged dataset
    data.dropna(inplace=True)
    
    return data
