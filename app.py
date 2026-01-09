from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Features expected by the model (excluding dropped ones)
FEATURES = [
    'volatile acidity', 'residual sugar', 'chlorides', 
    'total sulfur dioxide', 'density', 'sulphates', 'alcohol'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from form
        input_data = []
        for feature in FEATURES:
            val = float(request.form.get(feature))
            input_data.append(val)
        
        # Scale inputs
        input_df = pd.DataFrame([input_data], columns=FEATURES)
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][prediction]
        
        result = "Good" if prediction == 1 else "Bad"
        result_class = "result-good" if prediction == 1 else "result-bad"
        
        return render_template('index.html', 
                               prediction_text=f"Quality Prediction: {result}",
                               probability_text=f"Confidence: {probability*100:.1f}%",
                               result_class=result_class,
                               features=request.form.to_dict())
    
    except Exception as e:
        return render_template('index.html', error_text=f"Error in input: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
