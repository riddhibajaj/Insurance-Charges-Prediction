import streamlit as st
import pandas as pd
import numpy as np
import joblib

# defining the clip_bmi function
df_train = pd.read_csv(
    "insurance.csv")
bmi_upper_limit = df_train['bmi'].quantile(0.99)


def clip_bmi(x):
    return x.clip(upper=bmi_upper_limit)


# Loading the pipeline
pipeline = joblib.load(
    "model_pipeline.pkl")

# building the calculator
st.title("Insurance Charges Calculator")

# User inputs
age = st.number_input("Age", min_value=0, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker?", ["yes", "no"])
region = st.selectbox(
    "Region", ["northeast", "northwest", "southeast", "southwest"])
children = st.number_input(
    "Number of Children", min_value=0, max_value=10, value=0)

if st.button("Predict Charges"):
    df_input = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'sex': [sex],
        'smoker': [smoker],
        'region': [region],
        'children': [children]
    })

    log_pred = pipeline.predict(df_input)
    charges_pred = np.exp(log_pred)

    st.success(f"Predicted Insurance Charges: ${charges_pred[0]:.2f}")
