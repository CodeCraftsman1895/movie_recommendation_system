from .data_loading import load_and_merge_data
from .preprocessing import preprocess_data
from .model_training import train_and_save_model
import os

def generate_models_if_missing():
    """
    Checks if necessary .pkl files are missing, 
    and drives the overall ML pipeline organically if needed.
    """
    # Assuming that config paths have been initialized globally during imports
    from .config import DATA_PKL, SIMILARITY_PKL

    if not os.path.exists(DATA_PKL) or not os.path.exists(SIMILARITY_PKL):
        print("Missing .pkl artifacts. Initiating Data Pipeline to build them...")
        
        # 1. Loading Module
        raw_dataframe = load_and_merge_data()

        # 2. Preprocessing Module
        processed_dataframe = preprocess_data(raw_dataframe)

        # 3. Generating Vectorizer & Weights Pipeline 
        train_and_save_model(processed_dataframe)
    else:
        print("Required .pkl files natively detected. Skipping generation step.")

if __name__ == '__main__':
    generate_models_if_missing()
