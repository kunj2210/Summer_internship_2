import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

DATA_URL = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
DATA_PATH = "diabetes.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

def download_data():
    if not os.path.exists(DATA_PATH):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    else:
        print("Dataset already exists.")

def train():
    df = pd.read_csv(DATA_PATH)
    cols_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_clean] = df[cols_to_clean].replace(0, np.nan)
    
    # Impute missing values with their median
    for col in cols_to_clean:
        df[col] = df[col].fillna(df[col].median())
        
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature scaling is required for Logistic Regression to perform optimally
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    print(f"Train.py Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    
    # Export model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Model and Scaler successfully saved.")

if __name__ == "__main__":
    download_data()
    train()
