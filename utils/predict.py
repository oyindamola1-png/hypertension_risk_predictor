import joblib
import pandas as pd

# Load the best trained model
with open("./model/best_hypertension_model.pkl", "rb") as f:
    model = joblib.load("./model/best_hypertension_model.pkl")


# Load the corresponding scaler
with open("./model/scaler.pkl", "rb") as f:
    scaler = joblib.load("./model/scaler.pkl")

# The features used for training (excluding the target)
features = ['age', 'sex', 'cp', 'trestbps', 'chol',
            'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
            'slope', 'ca', 'thal', 'Outcome']

def preprocess_input(raw_data):
    """
    Map input fields directly to model features.
    """
    mapped = {
        'age': raw_data.get('age', 0),
        'sex': raw_data.get('sex', 0),
        'cp': raw_data.get('cp', 0),
        'trestbps': raw_data.get('systolic_bp', 0),
        'chol': raw_data.get('cholesterol', 0),
        'fbs': 1 if raw_data.get('glucose', 0) > 0 else 0,
        'restecg': raw_data.get('restecg', 0),
        'thalach': raw_data.get('heart_rate', 0),
        'exang': raw_data.get('smoking', 0),
        'oldpeak': raw_data.get('oldpeak', 0),
        'slope': raw_data.get('slope', 0),
        'ca': raw_data.get('alcohol_intake', 0),
        'thal': raw_data.get('physical_activity', 0),
        'Outcome': 0  # hardcoded as discussed
    }

    df = pd.DataFrame([mapped], columns=features)
    df.fillna(0, inplace=True)
    return scaler.transform(df)



def make_prediction(data_dict):
    """
    Preprocess the input and return prediction from the trained model.
    """
    processed_input = preprocess_input(data_dict)
    prediction = model.predict(processed_input)
    return int(prediction[0])  # Return as plain integer (0 or 1)