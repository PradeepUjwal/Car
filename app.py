
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and encoding
with open('xg_final_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('make_encoding.pkl', 'rb') as encoding_file:
    make_encoding = pickle.load(encoding_file)

# App title
st.title("Car Price Prediction App")

# Input fields
make = st.selectbox("Select Car Make", options=list(make_encoding.index))
condition = st.slider("Condition (1 to 5)", 1, 5, 3)
odometer = st.number_input("Odometer (in miles)", min_value=0, value=50000)
vehicle_age = st.number_input("Vehicle Age (in years)", min_value=0, value=5)
body_sedan = st.selectbox("Is it a Sedan?", options=["Yes", "No"])

# Encoding inputs
make_encoded = make_encoding[make]
body_sedan_encoded = 1 if body_sedan == "Yes" else 0

# Predict button
if st.button("Predict Selling Price"):
    features = np.array([[condition, odometer, vehicle_age, make_encoded, body_sedan_encoded]])
    prediction = model.predict(features)
    st.success(f"Predicted Selling Price: ${prediction[0]:,.2f}")

# Footer
st.markdown("---")
st.markdown("Created by Pradeep,Vivek,Sahithi,Spadana - LET'S GO 🚗💨")



