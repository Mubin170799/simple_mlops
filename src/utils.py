# src/utils.py
import pandas as pd

def load_data(file_path):
    # Load and return preprocessed data
    data = pd.read_csv(file_path)
    # Add any additional preprocessing steps if needed
    return data
