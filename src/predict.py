# src/predict.py
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import load_data

def predict_new_data(new_data):
    # Load trained model
    model = joblib.load('models/trained_model.pkl')

    # Load and preprocess new data
    new_data_df = pd.DataFrame(new_data, columns=['feature1', 'feature2', 'feature3'])
    X_new = new_data_df  # Adjust based on your features

    # Make predictions
    predictions = model.predict(X_new)

    # Decode labels
    label_encoder = LabelEncoder()
    predictions = label_encoder.inverse_transform(predictions)

    return predictions.tolist()

if __name__ == '__main__':
    new_data_example = [[1.0, 2.0, 3.0]]  # Replace with your new data
    predictions = predict_new_data(new_data_example)
    print(f'Predictions: {predictions}')
