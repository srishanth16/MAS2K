import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# Load trained model and label encoder
model = joblib.load("disease_prediction_model_tuned.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define the list of symptoms based on the training dataset
symptom_list = ['high_fever', 'breathlessness', 'throat_irritation', 'diarrhoea',
                'cough', 'fatigue', 'headache', 'nausea', 'muscle_pain']

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get("symptoms", [])

        # Convert symptoms to numerical format
        input_features = [1 if symptom in symptoms else 0 for symptom in symptom_list]
        
        # Ensure it matches the feature shape expected by the model
        input_array = np.array([input_features])  # Convert to numpy array

        # Predict disease
        prediction = model.predict(input_array)
        disease = label_encoder.inverse_transform(prediction)[0]  # Convert number back to disease name

        return jsonify({"predicted_disease": disease})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
