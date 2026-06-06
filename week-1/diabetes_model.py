import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# Step 1: Import Dataset
# ==========================================
print("--- Step 1: Importing Dataset ---")
# Load the diabetes dataset. 
# If it's not downloaded yet, we can load it directly from the URL.
url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
try:
    df = pd.read_csv("diabetes.csv")
    print("Loaded dataset from local 'diabetes.csv'.")
except FileNotFoundError:
    print("Local file not found. Downloading and loading dataset from URL...")
    df = pd.read_csv(url)
    df.to_csv("diabetes.csv", index=False)
    print("Dataset loaded and saved locally.")

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("First 5 rows of the dataset:")
print(df.head())


# ==========================================
# Step 2: Clean the Data
# ==========================================
print("\n--- Step 2: Cleaning the Data ---")

# In this dataset, some columns cannot physically be 0 (Glucose, BloodPressure, SkinThickness, Insulin, BMI).
# Zeros in these columns represent missing values.
cols_with_missing_data = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print("Checking for missing/invalid (0) values in physical columns:")
for col in cols_with_missing_data:
    zero_count = (df[col] == 0).sum()
    print(f" - {col}: {zero_count} invalid zero values")

# Replace 0 with NaN (Not a Number) to represent missing data
df[cols_with_missing_data] = df[cols_with_missing_data].replace(0, np.nan)

# Check for actual null/NaN values
print("\nMissing values (NaN) before cleaning:")
print(df.isnull().sum())

# Fill missing/NaN values with the average (mean) value of each column
print("\nCleaning missing values by filling them with the column averages (mean values)...")
for col in cols_with_missing_data:
    mean_value = df[col].mean()
    df[col] = df[col].fillna(mean_value)

print("\nMissing values (NaN) after cleaning:")
print(df.isnull().sum())


# ==========================================
# Step 3: Train the Model
# ==========================================
print("\n--- Step 3: Training the Model ---")

# Separate features (health details) and target (Outcome: 1 for Yes, 0 for No)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the dataset into Training set (80%) and Test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Initialize and train the Logistic Regression Model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")


# ==========================================
# Step 4: Test the Model
# ==========================================
print("\n--- Step 4: Testing the Model ---")

# Predict outcomes using the test data
y_pred = model.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Prediction Accuracy on Test Data: {accuracy * 100:.2f}%")

# Print a few example predictions vs actual values
print("\nExample Predictions vs Actual Values:")
results_comparison = pd.DataFrame({
    'Actual Value (Had Diabetes)': y_test.values,
    'Predicted Value': y_pred
})
print(results_comparison.head(10))
