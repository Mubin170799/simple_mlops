# generate_data.py
import pandas as pd
from sklearn.datasets import make_classification

def generate_synthetic_data():
    # Generate synthetic binary classification data
    X, y = make_classification(
        n_samples=1000, n_features=3, n_informative=3, n_redundant=0,
        n_clusters_per_class=1, random_state=42
    )

    # Create a DataFrame
    columns = ['feature1', 'feature2', 'feature3']
    df = pd.DataFrame(X, columns=columns)

    # Add binary labels (0 or 1)
    df['label'] = y

    # Save the dataset
    df.to_csv('data/processed/data.csv', index=False)

if __name__ == '__main__':
    generate_synthetic_data()
