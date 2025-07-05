from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained models and scaler
clf_model = joblib.load("diagnosis_model.pkl")
reg_model = joblib.load("recovery_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define the order of input features
feature_order = [
    'Age', 'Gender', 'Heart_Rate', 'Temperature', 'Systolic', 'Diastolic',
    'X-ray_Results', 'Lab_Test_Results', 'FamilyHistory', 'Allergies',
    'Hypertension_Risk', 'Treatment_Days'
]

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/analytics')
def analytics():
    return render_template("analytics.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form

        # Convert form data into proper types
        input_data = {
            'Age': int(data['age']),
            'Gender': int(data['gender']),  # 0=Male, 1=Female
            'Heart_Rate': float(data['heart_rate']),
            'Temperature': float(data['temperature']),
            'Systolic': float(data['systolic']),
            'Diastolic': float(data['diastolic']),
            'X-ray_Results': int(data['xray']),  # 0=Normal, 1=Abnormal
            'Lab_Test_Results': float(data['lab']),
            'FamilyHistory': int(data['family_history']),  # 0 or 1
            'Allergies': int(data['allergies']),  # 0 or 1
            'Treatment_Days': float(data['treatment_days'])
        }

        # Add derived feature
        input_data['Hypertension_Risk'] = int(
            input_data['Systolic'] >= 130 or input_data['Diastolic'] >= 80
        )

        # Arrange in correct order
        input_array = np.array([input_data[feat] for feat in feature_order]).reshape(1, -1)

        # Scale input
        scaled_input = scaler.transform(input_array)

        # Make predictions
        diagnosis = clf_model.predict(scaled_input)[0]
        recovery_days = reg_model.predict(scaled_input)[0]

        return render_template(
            "webpage.html",
            diagnosis=diagnosis,
            recovery=round(recovery_days, 1),
            input=input_data
        )

    except Exception as e:
        return render_template("webpage.html", error=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
