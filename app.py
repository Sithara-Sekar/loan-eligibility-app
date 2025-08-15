#!/usr/bin/env python3
"""
Streamlit app to serve a pre-trained scikit-learn model for Loan Eligibility prediction.
- Loads leader_model.pkl (saved locally).
- Lets users enter features manually or upload a CSV for batch scoring.
"""

import os
import pandas as pd
import streamlit as st
import joblib

APP_TITLE = "🏦 Loan Eligibility (Java-Free Version)"
MODEL_PATH = os.environ.get("MODEL_PATH", "leader_model.pkl")

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Could not find '{MODEL_PATH}'. Train the model first and save it as leader_model.pkl.")
    model = joblib.load(MODEL_PATH)
    return model

def predict_df(model, df: pd.DataFrame):
    preds = model.predict(df)
    proba = None
    try:
        proba = model.predict_proba(df)
    except AttributeError:
        pass
    preds_df = pd.DataFrame({"predict": preds})
    if proba is not None:
        for i, class_label in enumerate(model.classes_):
            preds_df[f"p_{class_label}"] = proba[:, i]
    return preds_df

def main():
    st.set_page_config(page_title="Loan Eligibility", layout="wide")
    st.title(APP_TITLE)
    st.caption("Powered by a pre-trained model. Enter features below or upload a CSV to get predictions.")

    # Load the model
    model = load_model()

    # Sidebar: batch scoring
    st.sidebar.header("📦 Batch scoring")
    batch_file = st.sidebar.file_uploader(
        "Upload CSV with the same feature columns as training data (excluding target).", type=["csv"]
    )
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        st.sidebar.write("Preview:", batch_df.head())
        if st.sidebar.button("Run batch predictions"):
            preds = predict_df(model, batch_df)
            out = pd.concat([batch_df.reset_index(drop=True), preds], axis=1)
            st.write("Batch predictions:", out.head(20))
            st.download_button("Download predictions.csv", data=out.to_csv(index=False), file_name="predictions.csv")

    st.divider()
    st.subheader("🧮 Single prediction")

    # Categorical inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self_Employed", ["Yes", "No"])
    property_area = st.selectbox("Property_Area", ["Urban", "Semiurban", "Rural"])

    # Numeric inputs
    applicant_income = st.number_input("ApplicantIncome", min_value=0, value=5000, step=100)
    coapplicant_income = st.number_input("CoapplicantIncome", min_value=0, value=0, step=100)
    loan_amount = st.number_input("LoanAmount (in thousands)", min_value=0, value=128, step=1)
    loan_amount_term = st.number_input("Loan_Amount_Term (in days)", min_value=12, value=360, step=12)
    credit_history = st.selectbox("Credit_History", [1.0, 0.0])

    # Assemble row
    row = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": float(credit_history),
        "Property_Area": property_area,
    }
    input_df = pd.DataFrame([row])

    if st.button("Predict eligibility"):
        preds = predict_df(model, input_df)
        st.metric("Prediction", preds.iloc[0]["predict"])
        st.write("Raw prediction output:", preds)

    with st.expander("Show input row as DataFrame"):
        st.dataframe(input_df)

    st.info("Tip: Ensure leader_model.pkl was trained with the same schema as the UI fields or the uploaded CSV.")

if __name__ == "__main__":
    main()
