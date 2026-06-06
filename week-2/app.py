import os
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model assets
LR_MODEL_PATH = "linear_reg_model.pkl"
RF_MODEL_PATH = "random_forest_model.pkl"
SCALER_PATH = "scaler.pkl"
IMPUTER_PATH = "imputer.pkl"
MAPPINGS_PATH = "mappings.pkl"

required_files = [LR_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH, IMPUTER_PATH, MAPPINGS_PATH]
if any(not os.path.exists(f) for f in required_files):
    raise FileNotFoundError("Model assets not found. Run 'train.py' first.")

lr_model = joblib.load(LR_MODEL_PATH)
rf_model = joblib.load(RF_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
imputer_values = joblib.load(IMPUTER_PATH)
mappings = joblib.load(MAPPINGS_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # 1. Extract values and handle missing data for mileage/engine
        try:
            year = int(data.get('Year', 2020))
        except ValueError:
            return jsonify({'error': 'Invalid year. Must be an integer.'}), 400
        
        # Mileage (handle missing)
        mileage_raw = data.get('Mileage')
        if mileage_raw is None or str(mileage_raw).strip() == "":
            mileage = float(imputer_values['Mileage'])
            mileage_imputed = True
        else:
            try:
                mileage = float(mileage_raw)
                if mileage < 0:
                    return jsonify({'error': 'Mileage cannot be negative.'}), 400
                mileage_imputed = False
            except ValueError:
                return jsonify({'error': 'Invalid mileage. Must be a number.'}), 400

        # Engine Size (handle missing)
        engine_raw = data.get('Engine_Size')
        if engine_raw is None or str(engine_raw).strip() == "":
            engine_size = float(imputer_values['Engine_Size'])
            engine_imputed = True
        else:
            try:
                engine_size = float(engine_raw)
                if engine_size <= 0:
                    return jsonify({'error': 'Engine size must be greater than 0.'}), 400
                engine_imputed = False
            except ValueError:
                return jsonify({'error': 'Invalid engine size. Must be a number.'}), 400

        # Categorical mapping & validation
        fuel_type = data.get('Fuel_Type', 'Petrol')
        owner_type = data.get('Owner_Type', 'First-hand')
        car_condition = data.get('Car_Condition', 'Good')
        
        if fuel_type not in mappings['fuel']:
            return jsonify({'error': f"Invalid fuel type: '{fuel_type}'. Choose from {list(mappings['fuel'].keys())}."}), 400
        if owner_type not in mappings['owner']:
            return jsonify({'error': f"Invalid owner type: '{owner_type}'. Choose from {list(mappings['owner'].keys())}."}), 400
        if car_condition not in mappings['condition']:
            return jsonify({'error': f"Invalid car condition: '{car_condition}'. Choose from {list(mappings['condition'].keys())}."}), 400
            
        fuel_num = mappings['fuel'][fuel_type]
        owner_num = mappings['owner'][owner_type]
        condition_num = mappings['condition'][car_condition]
        
        # Create DataFrame with the exact features in training order
        feature_df = pd.DataFrame([{
            'Year': year,
            'Mileage': mileage,
            'Engine_Size': engine_size,
            'Fuel_Type_Num': fuel_num,
            'Owner_Type_Num': owner_num,
            'Car_Condition_Num': condition_num
        }])
        
        # Scale features
        features_scaled = scaler.transform(feature_df)
        
        # Predict using both models
        pred_lr = float(lr_model.predict(features_scaled)[0])
        pred_rf = float(rf_model.predict(features_scaled)[0])
        
        # Ensure predicted price is non-negative
        pred_lr = max(0.1, pred_lr)
        pred_rf = max(0.1, pred_rf)
        
        # Prepare response & recommendations based on predictions and condition
        recommendations = []
        if car_condition == 'Poor':
            recommendations.append("The car's condition is poor. Investing in minor repairs (engine tuning, denting/painting) could raise its resale value.")
        if mileage > 100000:
            recommendations.append("The mileage exceeds 100,000 km. Regular servicing records will help justify a higher price to potential buyers.")
        if owner_type != 'First-hand':
            recommendations.append("Multiple prior owners decrease resale value. Ensure all registration documents are clear and updated to prevent additional discount demands.")
        if len(recommendations) == 0:
            recommendations.append("The car is in good condition with moderate mileage. Keeping it clean and polishing before inspection will help fetch the best price.")
            
        return jsonify({
            'linear_regression_price': round(pred_lr, 2),
            'random_forest_price': round(pred_rf, 2),
            'difference': round(abs(pred_lr - pred_rf), 2),
            'preferred_model': 'Random Forest Regressor',
            'mileage_imputed': mileage_imputed,
            'engine_imputed': engine_imputed,
            'imputed_mileage_val': round(mileage, 1) if mileage_imputed else None,
            'imputed_engine_val': round(engine_size, 1) if engine_imputed else None,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Running server locally on port 5000
    app.run(debug=True, host='127.0.0.1', port=5000)
