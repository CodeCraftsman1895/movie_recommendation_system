import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import DATA_PKL, SIMILARITY_PKL

def train_and_save_model(data):
    """
    Constructs TF-IDF features based on the unified 'tags' feature
    and evaluates memory-intensive Cosine Similarities.
    Outputs are persisted linearly as serialized binary files for Fast Load in UI.
    """
    print("Vectorizing Text via TF-IDF (Max Features: 5000)...")
    tfidf = TfidfVectorizer(max_features=5000)
    vectors = tfidf.fit_transform(data['tags']).toarray()

    print("Computing Cosine Similarity Metric Matrix...")
    similarity = cosine_similarity(vectors)

    print(f"Persisting serialized models to Disk...")
    # Using 'wb' standard writes
    with open(DATA_PKL, 'wb') as f:
        pickle.dump(data.to_dict(), f)

    with open(SIMILARITY_PKL, 'wb') as f:
        pickle.dump(similarity, f)

    print("Successfully built and persisted Recommendation Engines.")
