# Diabetes Prediction Model (InternPe Week 1 Task)

An interactive, end-to-end Machine Learning web application that predicts whether a patient has diabetes based on health vitals (Glucose level, BMI, Age, Blood Pressure, Insulin, etc.) using a **Logistic Regression** classifier.

This project is completed as part of the **InternPe Internship Program - Week 1 Task**.

---

## 📊 Dataset Description
The model is trained on the standard **Pima Indians Diabetes Dataset** (`diabetes.csv`). It contains clinical variables to predict the presence of diabetes:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-hour serum insulin (mu U/ml)
- **BMI**: Body Mass Index (weight in kg / (height in m)²)
- **DiabetesPedigreeFunction**: Genetic diabetes likelihood score based on family history
- **Age**: Patient age (years)
- **Outcome**: Target variable (`1` for Diabetes (Yes), `0` for No Diabetes (No))

---

## 🔧 Steps Implemented

### 1. Import Dataset
- Load the diabetes dataset (`diabetes.csv`) into Python using `pandas`.

### 2. Clean the Data
- Identifies invalid `0` values in clinical measurements (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) where a zero is physiologically impossible.
- Replaces those invalid zeros with `NaN` and fills (imputes) them using the **mean/average** value of each feature.

### 3. Train the Model
- Splits the dataset into a **Training set (80%)** and a **Test set (20%)**.
- Standardizes features using a `StandardScaler` to ensure the Logistic Regression algorithm is scale-invariant.
- Fits a **Logistic Regression** classifier (`sklearn.linear_model.LogisticRegression`).

### 4. Test the Model
- Evaluates model performance on the unseen test set.
- Prints the prediction accuracy score (approx. **70.78%** to **75.32%** depending on preprocessing details).
- Visualizes results in an interactive web application.

---

## 🖥️ Localhost Web Application
To demonstrate the model's capabilities in a real-world scenario, a premium web application dashboard is built on top of the model.

### Features:
- **Glassmorphism Design Theme**: Modern dark mode dashboard with glassy blur backdrops and ambient glowing elements.
- **Dynamic Risk Gauge**: Interactive SVG progress ring indicating risk probability (0% - 100%).
- **Color-Coded Badges**: Badge colors dynamically shift (Green for Low Risk, Orange for Moderate Risk, Red for High Risk) based on evaluation.
- **Clinical Recommendations**: Generates personalized lifestyle, diet, and clinical recommendations based on risk scores.

---

## 📁 Repository Folder Structure
```
├── templates/
│   └── index.html          # Frontend HTML structure for the web dashboard
├── static/
│   ├── style.css           # Premium glassmorphic styling sheet
│   └── app.js              # Javascript handler for async predictions & SVG animations
├── diabetes_model.py       # Standalone CLI python script (Imports, cleans, trains & tests)
├── train.py                # Python training pipeline that outputs serialization files
├── app.py                  # Flask server running on http://localhost:5000
├── diabetes.csv            # The dataset file
├── model.pkl               # Serialized Logistic Regression model file
├── scaler.pkl              # Serialized StandardScaler object file
├── requirements.txt        # Required python packages for installation
└── .gitignore              # Files to exclude from repository tracking
```

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone <your-github-repo-link>
cd <repository-folder>
```

### 2. Install Dependencies
Make sure you have Python 3 installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Standalone Script (CLI)
To check the basic model pipeline and output printouts:
```bash
python diabetes_model.py
```

### 4. Start the Web Dashboard
To start the Flask development server on localhost:
```bash
python app.py
```
Open your web browser and go to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**
