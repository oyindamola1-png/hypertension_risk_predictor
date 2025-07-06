from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from utils.predict import make_prediction

app = Flask(__name__)

CORS(app)


@app.route('/')
def home():
    return render_template('index.html', title="Hypertension Risk Prediction")

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
