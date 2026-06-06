import os
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or Scaler file not found. Run 'train.py' first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        required_features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 
            'SkinThickness', 'Insulin', 'BMI', 
            'DiabetesPedigreeFunction', 'Age'
        ]
        
        input_features = []
        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            try:
                val = float(data[feature])
                input_features.append(val)
            except ValueError:
                return jsonify({'error': f'Invalid value for {feature}. Must be a number.'}), 400
        
        features_arr = np.array(input_features).reshape(1, -1)
        features_scaled = scaler.transform(features_arr)
        
        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1]) * 100
        
        if probability < 30:
            risk_level = "Low"
            color = "#10B981"
            recommendations = [
                "Maintain your healthy lifestyle with balanced nutrition and regular physical activity.",
                "Continue tracking your glucose levels periodically during regular checkups.",
                "Keep hydrated and ensure a minimum of 7-8 hours of sleep per day."
            ]
        elif probability < 70:
            risk_level = "Moderate"
            color = "#F59E0B"
            recommendations = [
                "Consider refining your diet by reducing refined carbohydrates and sugar intake.",
                "Incorporate moderate exercise (e.g., brisk walking, cycling) for at least 150 minutes a week.",
                "Monitor your blood pressure and BMI closely.",
                "Consult with a healthcare provider for a routine checkup and blood tests (HbA1c)."
            ]
        else:
            risk_level = "High"
            color = "#EF4444"
            recommendations = [
                "We highly recommend scheduling a comprehensive consultation with a physician or endocrinologist.",
                "Request a fasting blood glucose test and a HbA1c test from your medical provider.",
                "Work with a nutritionist to design a structured low-glycemic, high-fiber meal plan.",
                "Closely monitor your activity levels, weight, and blood sugar levels."
            ]
            
        return jsonify({
            'prediction': prediction,
            'probability': round(probability, 2),
            'risk_level': risk_level,
            'color': color,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Running server locally on port 5000
    app.run(debug=True, host='127.0.0.1', port=5000)
