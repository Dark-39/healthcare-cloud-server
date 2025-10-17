from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the powerful model from the file in our repository
model = joblib.load('random_forest_model.joblib')

@app.route('/')
def home():
    return "Cloud Server is active and ready for analysis."

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get the patient data sent from the edge server
        patient_data = request.get_json()

        # Convert JSON to a pandas DataFrame for the model
        # We also need to add the one-hot encoded columns for 'thal'
        df = pd.DataFrame([patient_data])
        df['thal_normal'] = 0
        df['thal_reversible_defect'] = 0

        # Reorder columns to match the training data exactly
        training_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 
                            'thal_normal', 'thal_reversible_defect']
        df = df.reindex(columns=training_columns, fill_value=0)

        # Perform the deeper analysis
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)

        result = int(prediction[0])
        confidence = float(max(prediction_proba[0]))

        # Print the result to the Render logs for us to see
        print(f"Cloud analysis complete. Prediction: {result}, Confidence: {confidence:.4f}")

        return jsonify({"status": "success", "cloud_prediction": result}), 200

    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400