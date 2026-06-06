import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

def main():
    print("Starting Model Training Pipeline...")
    
    # 1. Load Dataset
    df = pd.read_csv('car_details.csv')
    
    # 2. Data Cleaning - Calculate and save medians for imputation
    mileage_median = df['Mileage'].median()
    engine_median = df['Engine_Size'].median()
    
    # Impute missing values
    df['Mileage'] = df['Mileage'].fillna(mileage_median)
    df['Engine_Size'] = df['Engine_Size'].fillna(engine_median)
    
    # 3. Categorical Mappings
    fuel_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
    owner_mapping = {'First-hand': 0, 'Second-hand': 1, 'Third-hand': 2}
    condition_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
    
    # Convert features to numbers
    df['Fuel_Type_Num'] = df['Fuel_Type'].map(fuel_mapping)
    df['Owner_Type_Num'] = df['Owner_Type'].map(owner_mapping)
    df['Car_Condition_Num'] = df['Car_Condition'].map(condition_mapping)
    
    # Drop original categoricals
    X = df.drop(['Selling_Price', 'Fuel_Type', 'Owner_Type', 'Car_Condition'], axis=1)
    y = df['Selling_Price']
    
    # 4. Standard Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. Train Models
    print("Training Linear Regression Model...")
    lr_model = LinearRegression()
    lr_model.fit(X_scaled, y)
    
    print("Training Random Forest Regressor Model...")
    rf_model = RandomForestRegressor(n_estimators=150, random_state=42)
    rf_model.fit(X_scaled, y)
    
    # 6. Save Artifacts
    print("Saving serialized model assets to disk...")
    joblib.dump(lr_model, 'linear_reg_model.pkl')
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save imputation values and mappings
    imputer_values = {
        'Mileage': mileage_median,
        'Engine_Size': engine_median
    }
    joblib.dump(imputer_values, 'imputer.pkl')
    
    mappings = {
        'fuel': fuel_mapping,
        'owner': owner_mapping,
        'condition': condition_mapping
    }
    joblib.dump(mappings, 'mappings.pkl')
    
    print("Model pipeline run successfully! All artifacts saved.")

if __name__ == '__main__':
    main()
