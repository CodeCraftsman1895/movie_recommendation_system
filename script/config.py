import os

# Base paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Input files
CREDITS_CSV = os.path.join(DATA_DIR, 'credits.csv')
KEYWORDS_CSV = os.path.join(DATA_DIR, 'keywords.csv')
MOVIES_METADATA_CSV = os.path.join(DATA_DIR, 'movies_metadata.csv')

# Output files
DATA_PKL = os.path.join(ROOT_DIR, 'data.pkl')
SIMILARITY_PKL = os.path.join(ROOT_DIR, 'similarity.pkl')
