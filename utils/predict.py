import pickle
import pandas as pd

# Load the model from .pkl
with open("./model/xgb_hypertension_model.pkl", "rb") as f:
    model = pickle.load(f)

def preprocess_input(data_dict):
    """
    Convert input dictionary into a pandas DataFrame with correct order of features.
    Update the 'features' list to match your training data's exact columns.
    """
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 
                'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'ca', 'thal', 'target']
    print (data_dict)
    print (features)

    input_df = pd.DataFrame([data_dict], columns=features)
    return input_df

def make_prediction(data_dict):
    """
    Process the input and return the model's prediction.
    """
    input_df = preprocess_input(data_dict)
    prediction = model.predict(input_df)
    return int(prediction[0])  # Ensure it's a plain integer
