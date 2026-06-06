import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset of 1000 cars
n_samples = 1000

# 1. Year of purchase: 2012 to 2024
years = np.random.randint(2012, 2025, size=n_samples)

# 2. Mileage (km driven): 5000 to 180000 km, correlated with age
# Newer cars tend to have lower mileage
base_mileage = (2025 - years) * np.random.randint(8000, 15000, size=n_samples)
mileage = np.clip(base_mileage + np.random.normal(0, 5000, size=n_samples), 5000, 200000).astype(float)

# 3. Fuel type: Petrol (60%), Diesel (35%), CNG (5%)
fuel_types = np.random.choice(['Petrol', 'Diesel', 'CNG'], size=n_samples, p=[0.60, 0.35, 0.05])

# 4. Engine size (cc): 800cc to 3000cc
# Let's define some standard engine sizes: 998 (1.0L), 1197 (1.2L), 1497 (1.5L), 1998 (2.0L), 2498 (2.5L), 2993 (3.0L)
engine_options = [998, 1197, 1497, 1998, 2498, 2993]
engine_probs = [0.25, 0.35, 0.20, 0.12, 0.05, 0.03]
engine_sizes = np.random.choice(engine_options, size=n_samples, p=engine_probs).astype(float)

# 5. Owner type: First-hand (70%), Second-hand (25%), Third-hand (5%)
owner_types = np.random.choice(['First-hand', 'Second-hand', 'Third-hand'], size=n_samples, p=[0.70, 0.25, 0.05])

# 6. Car condition: Excellent (20%), Good (50%), Fair (20%), Poor (10%)
car_conditions = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], size=n_samples, p=[0.20, 0.50, 0.20, 0.10])

# 7. Selling Price calculation (in Lakhs): base price + features influence + noise
# Base price by engine size (larger engine = higher class car)
base_prices = {998: 4.5, 1197: 6.5, 1497: 10.0, 1998: 18.0, 2498: 28.0, 2993: 45.0}
price = np.array([base_prices[e] for e in engine_sizes])

# Age depreciation: depreciates ~8% per year
age = 2025 - years
depreciation_factor = 0.92 ** age
price = price * depreciation_factor

# Mileage discount: deduct price based on mileage
price = price - (mileage / 100000) * 1.5

# Fuel type influence: Diesel is slightly more expensive, CNG is slightly cheaper
fuel_multipliers = {'Petrol': 1.0, 'Diesel': 1.15, 'CNG': 0.85}
price = price * np.array([fuel_multipliers[f] for f in fuel_types])

# Owner type discount: Second-hand is 15% cheaper, Third-hand is 30% cheaper
owner_multipliers = {'First-hand': 1.0, 'Second-hand': 0.85, 'Third-hand': 0.70}
price = price * np.array([owner_multipliers[o] for o in owner_types])

# Car condition influence
condition_multipliers = {'Excellent': 1.1, 'Good': 1.0, 'Fair': 0.8, 'Poor': 0.6}
price = price * np.array([condition_multipliers[c] for c in car_conditions])

# Add Gaussian noise
price = price + np.random.normal(0, 0.5, size=n_samples)

# Ensure selling price is realistic (minimum of 0.8 Lakhs)
price = np.clip(price, 0.8, 95.0)

# Introduce 5% missing values (NaN) in Mileage and Engine Size to simulate real-world data cleaning requirements
nan_mask_mileage = np.random.rand(n_samples) < 0.05
nan_mask_engine = np.random.rand(n_samples) < 0.05

mileage[nan_mask_mileage] = np.nan
engine_sizes[nan_mask_engine] = np.nan

# Construct DataFrame
df = pd.DataFrame({
    'Year': years,
    'Mileage': mileage,
    'Fuel_Type': fuel_types,
    'Engine_Size': engine_sizes,
    'Owner_Type': owner_types,
    'Car_Condition': car_conditions,
    'Selling_Price': np.round(price, 2)
})

# Save to CSV
output_path = 'car_details.csv'
df.to_csv(output_path, index=False)
print(f"Dataset generated successfully! Saved to '{output_path}'")
print(f"Total samples: {len(df)}")
print(f"Missing values in Mileage: {df['Mileage'].isnull().sum()}")
print(f"Missing values in Engine Size: {df['Engine_Size'].isnull().sum()}")
print("\nDataset Sample:")
print(df.head())
