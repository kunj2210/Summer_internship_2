import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_and_clean_data(filepath):
    print("Step 1: Loading Dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    
    print("\nStep 2: Data Cleaning & Preprocessing...")
    
    # 2.1 Handle missing mileage/engine data using Median imputation
    print("Checking for missing values:")
    print(df.isnull().sum())
    
    mileage_median = df['Mileage'].median()
    engine_median = df['Engine_Size'].median()
    
    print(f"Imputing missing Mileage with median: {mileage_median:.1f} km")
    print(f"Imputing missing Engine Size with median: {engine_median:.1f} cc")
    
    df['Mileage'] = df['Mileage'].fillna(mileage_median)
    df['Engine_Size'] = df['Engine_Size'].fillna(engine_median)
    
    # 2.2 Convert fuel type and other categorical features to numbers
    # We will use explicit mapping to ensure numerical order makes sense (especially for condition/owner)
    fuel_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
    owner_mapping = {'First-hand': 0, 'Second-hand': 1, 'Third-hand': 2}
    condition_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
    
    print("\nMapping categorical features to numbers:")
    print(f"Fuel Type mapping: {fuel_mapping}")
    print(f"Owner Type mapping: {owner_mapping}")
    print(f"Car Condition mapping: {condition_mapping}")
    
    df['Fuel_Type_Num'] = df['Fuel_Type'].map(fuel_mapping)
    df['Owner_Type_Num'] = df['Owner_Type'].map(owner_mapping)
    df['Car_Condition_Num'] = df['Car_Condition'].map(condition_mapping)
    
    # Drop original categorical columns
    df_processed = df.drop(['Fuel_Type', 'Owner_Type', 'Car_Condition'], axis=1)
    
    return df_processed, fuel_mapping, owner_mapping, condition_mapping

def train_and_evaluate():
    # Load and clean
    df, fuel_map, owner_map, cond_map = load_and_clean_data('car_details.csv')
    
    # Define features and target
    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']
    
    print("\nFeatures used for training:")
    print(list(X.columns))
    
    # 3. Train/Test Split (80% train, 20% test)
    print("\nStep 3: Splitting data into train/test sets (80% / 20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 2.3 Scale features
    print("\nStep 3.1: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Choose Models
    print("\nStep 4: Initializing models...")
    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # 5. Train the Models
    print("Step 5: Training models...")
    lr_model.fit(X_train_scaled, y_train)
    rf_model.fit(X_train_scaled, y_train)
    
    # 6. Test the Models
    print("Step 6: Testing and evaluating models...")
    
    # Predictions
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_rf = rf_model.predict(X_test_scaled)
    
    # Evaluate Linear Regression
    lr_r2 = r2_score(y_test, y_pred_lr)
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    
    # Evaluate Random Forest
    rf_r2 = r2_score(y_test, y_pred_rf)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    
    print("\n--- Model Performance Comparison ---")
    print(f"Linear Regression:")
    print(f"  R² Score (Accuracy): {lr_r2:.4f} ({lr_r2*100:.2f}%)")
    print(f"  Mean Absolute Error: {lr_mae:.4f} Lakhs")
    print(f"  Root Mean Squared Error: {lr_rmse:.4f} Lakhs")
    
    print(f"Random Forest Regressor:")
    print(f"  R² Score (Accuracy): {rf_r2:.4f} ({rf_r2*100:.2f}%)")
    print(f"  Mean Absolute Error: {rf_mae:.4f} Lakhs")
    print(f"  Root Mean Squared Error: {rf_rmse:.4f} Lakhs")
    
    # 7. Make Predictions (Demo)
    print("\nStep 7: Making sample predictions...")
    # Sample car: Year=2020, Mileage=35000, Fuel=Petrol(0), Engine=1497, Owner=First-hand(0), Condition=Good(2)
    sample_car = pd.DataFrame([{
        'Year': 2020,
        'Mileage': 35000.0,
        'Engine_Size': 1497.0,
        'Fuel_Type_Num': 0,
        'Owner_Type_Num': 0,
        'Car_Condition_Num': 2
    }])
    
    sample_scaled = scaler.transform(sample_car)
    
    pred_lr = lr_model.predict(sample_scaled)[0]
    pred_rf = rf_model.predict(sample_scaled)[0]
    
    print("\nSample Car Details:")
    print("  Year: 2020 | Mileage: 35,000 km | Fuel: Petrol | Engine: 1497cc | Owner: First-hand | Condition: Good")
    print(f"  Predicted Price (Linear Regression): {pred_lr:.2f} Lakhs")
    print(f"  Predicted Price (Random Forest): {pred_rf:.2f} Lakhs")

if __name__ == "__main__":
    train_and_evaluate()
