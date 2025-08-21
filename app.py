# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 23:55:11 2025

@author: lalit
"""

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
try:
    model = joblib.load('logistic_regression_model.pkl')
except FileNotFoundError:
    st.error("Model file 'logistic_regression_model.pkl' not found. Please train and save the model first.")
    st.stop()

st.set_page_config(page_title="Titanic Survival Predictor")
st.title("🚢 Titanic Survival Predictor")
st.write("Enter the passenger's details to predict their survival probability.")

# Create user input widgets
pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2) # Default to 3rd class
sex_map = {'Male': 1, 'Female': 0}
sex = st.selectbox("Sex", ['Male', 'Female'])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=6, value=0)
fare = st.number_input("Fare (USD)", min_value=0.0, value=30.0)
embarked_map = {'Southampton': 2, 'Cherbourg': 0, 'Queenstown': 1}
embarked = st.selectbox("Port of Embarkation", ['Southampton', 'Cherbourg', 'Queenstown'])

# Prepare user input as a DataFrame
user_data = {
    'Pclass': pclass,
    'Sex': sex_map[sex],
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked_map[embarked]
}

features_df = pd.DataFrame([user_data])

# Make prediction on button click
if st.button("Predict Survival"):
    prediction = model.predict(features_df)
    prediction_proba = model.predict_proba(features_df)[:, 1]

    # Display results
    if prediction[0] == 1:
        st.success(f"🎉 **Prediction:** The passenger would likely have survived!")
        st.balloons()
        st.metric("Survival Probability", f"{prediction_proba[0]:.2%}")
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/07/Titanic_survivors_at_the_scene_of_the_disaster.jpg", caption="Survivors of the Titanic disaster", use_column_width=True)
    else:
        st.error(f"😔 **Prediction:** The passenger would likely not have survived.")
        st.metric("Survival Probability", f"{prediction_proba[0]:.2%}")
        st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", caption="RMS Titanic", use_column_width=True)
