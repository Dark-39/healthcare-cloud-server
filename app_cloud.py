from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# --- Load the necessary files ---
try:
    model = joblib.load('cloud_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Cloud model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {e}")

# This list MUST exactly match the columns from train_models.py
training_columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 
    'thal_1', 'thal_2', 'thal_3'
]

@app.route('/')
def home():
    return "Cloud Server is active and ready for analysis."

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        patient_data = request.get_json()
        print(f"Received data for analysis: {patient_data}")
        
        df = pd.DataFrame([patient_data])
        
        # --- THE CRITICAL FIX ---
        # Process the incoming data EXACTLY like the training data
        df['thal'] = df['thal'].astype('category')
        df = pd.get_dummies(df) 
        df = df.reindex(columns=training_columns, fill_value=0)
        
        # Scale the data
        scaled_data = scaler.transform(df)
        
        # Make the prediction
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)
        
        result = int(prediction[0])
        confidence = float(max(prediction_proba[0]))
        
        print(f"Cloud analysis complete. Prediction: {result}, Confidence: {confidence:.4f}")
        
        return jsonify({"status": "success", "cloud_prediction": result}), 200

    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400