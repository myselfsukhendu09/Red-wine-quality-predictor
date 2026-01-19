
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(page_title="Red Wine Quality Predictor", page_icon="🍷")

st.title("🍷 Red Wine Quality Prediction")
st.markdown("---")

# Load model and scaler
if os.path.exists("model.pkl") and os.path.exists("scaler.pkl"):
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
else:
    st.error("Model files not found!")
    st.stop()

st.sidebar.header("Input Wine Features")

def user_input_features():
    fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 7.4)
    volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 1.6, 0.7)
    citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.0)
    residual_sugar = st.sidebar.slider("Residual Sugar", 0.9, 15.5, 1.9)
    chlorides = st.sidebar.slider("Chlorides", 0.01, 0.6, 0.076)
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 1, 72, 11)
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 6, 289, 34)
    density = st.sidebar.slider("Density", 0.990, 1.004, 0.9978)
    ph = st.sidebar.slider("pH", 2.7, 4.0, 3.51)
    sulphates = st.sidebar.slider("Sulphates", 0.3, 2.0, 0.56)
    alcohol = st.sidebar.slider("Alcohol", 8.4, 14.9, 9.4)
    
    data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': ph,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    return pd.DataFrame(data, index=[0])

df = user_input_features()

st.subheader("User Input parameters")
st.write(df)

if st.button("Predict Quality"):
    scaled_df = scaler.transform(df)
    prediction = model.predict(scaled_df)
    
    st.subheader("Prediction")
    st.success(f"The predicted quality of the wine is: {prediction[0]}")
    
    if prediction[0] >= 7:
        st.write("✨ This is considered **Good Quality** wine!")
    else:
        st.write("🍷 This is considered **Average/Poor Quality** wine.")
