from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.predict import make_prediction

app = Flask(__name__)

CORS(app) # Enable CORS for all routes


@app.route('/')
def home():
    return "🏥 Hypertension Risk Prediction API is up and running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        prediction = make_prediction(data)
        result = {
            "prediction": prediction,
            "risk_level": "High" if prediction == 1 else "Low"
        }
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
