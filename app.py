import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model and preprocessors
with open('insurance_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

st.title("Insurance Charges Prediction")

st.write("Provide the required details:")

# User inputs
age = st.number_input("Age", min_value=0, max_value=120, value=30)
sex = st.selectbox("Sex", options=["female", "male"])
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", options=["no", "yes"])
region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict Charges"):
    # Prepare the input as a DataFrame
    input_data = {
        "age": [age],
        "sex": [0 if sex == "female" else 1],
        "bmi": [bmi],
        "children": [children],
        "smoker": [1 if smoker == "yes" else 2],  # Adjusted to match your encoding
        "region": [region]
    }
    input_df = pd.DataFrame(input_data)
    
    # Define numerical columns
    numerical_cols = ['age', 'bmi', 'children', 'sex', 'smoker']
    
    # Scale numerical features
    numerical_data = scaler.transform(input_df[numerical_cols])
    scaled_numerical_df = pd.DataFrame(numerical_data, columns=numerical_cols)
    
    # Encode region
    region_data = encoder.transform(input_df[['region']])
    region_df = pd.DataFrame(
        region_data,
        columns=encoder.get_feature_names_out(['region'])
    )
    
    # Combine features
    X_combined = pd.concat([scaled_numerical_df, region_df], axis=1)
    
    # Predict the charges using the loaded model
    prediction = model.predict(X_combined)
    
    st.success(f"Predicted Insurance Charges: ${prediction[0]:.2f}")