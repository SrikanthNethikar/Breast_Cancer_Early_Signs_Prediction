import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Load the trained model
with open("breast_cancer_early_signs_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page title
st.title("🩺 Breast Cancer Early Signs Prediction")

# Input form
st.header("Enter Patient Information (40 Features)")

# Replace with your actual 40 feature names if available
feature_names = [f"Feature {i+1}" for i in range(40)]

# Collect inputs
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

# Predict button
if st.button("Predict Risk"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.success(f"Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
    st.info(f"Risk Probability Score: {proba:.2f}")

    # SHAP Explanation
    st.subheader("Feature Contribution (SHAP Waterfall Plot)")
    explainer = shap.Explainer(model.predict_proba, input_df)
    shap_values = explainer(input_df)

    shap_val = shap_values.values[0][:]      # (40,)
    expected_val = explainer.expected_value[1]  # class 1

    fig = shap.plots._waterfall.waterfall_legacy(
        expected_val,
        shap_val,
        feature_names=feature_names,
        max_display=10
    )
    st.pyplot(fig)