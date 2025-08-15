#!/usr/bin/env python3
"""
Streamlit app to serve a pre-trained scikit-learn model for Loan Eligibility prediction.
- Loads models/leader_model.pkl (saved locally).
- Ensures preprocessing matches training (one-hot encoding + numeric features).
"""

import os
import pandas as pd
import streamlit as st
import joblib

APP_TITLE = "🏦 Loan Eligibility Prediction"
MODEL_PATH = os.path.join("models", "leader_model.pkl")

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Could not find '{MODEL_PATH}'. Please ensure the model file exists.")
        raise FileNotFoundError(f"Could not find '{MODEL_PATH}'. Train the model first.")
    try:
        obj = joblib.load(MODEL_PATH)
        if isinstance(obj, dict):
            model = obj.get("model")  # Adjust key based on your pickle file’s structure
            if model is None:
                raise ValueError("No scikit-learn model found in the dictionary.")
        else:
            model = obj
        if not hasattr(model, "predict"):
            raise ValueError("Loaded object is not a valid scikit-learn model.")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        raise

def preprocess_input(df: pd.DataFrame):
    """Preprocess input to match training schema (one-hot encoding for categoricals)."""
    try:
        df = df.copy()
        valid_dependents = ["0", "1", "2", "3+"]
        if not df["Dependents"].isin(valid_dependents).all():
            raise ValueError(f"Dependents must be one of: {', '.join(valid_dependents)}")
        df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

        # One-hot encode categorical columns
        cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
        df = pd.get_dummies(df, columns=cat_cols)

        # Use model’s feature_names_in_ if available, else fallback to hardcoded list
        model = load_model()
        if hasattr(model, "feature_names_in_"):
            model_columns = model.feature_names_in_
        else:
            st.warning("Model lacks feature_names_in_. Using hardcoded feature names.")
            model_columns = [
                "Dependents", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
                "Credit_History", "Gender_Female", "Gender_Male", "Married_No", "Married_Yes",
                "Education_Graduate", "Education_Not Graduate", "Self_Employed_No", "Self_Employed_Yes",
                "Property_Area_Rural", "Property_Area_Semiurban", "Property_Area_Urban"
            ]
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[model_columns]
        return df
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        raise

def predict_df(model, df: pd.DataFrame):
    try:
        df_processed = preprocess_input(df)
        preds = model.predict(df_processed)
        proba = None
        try:
            proba = model.predict_proba(df_processed)
        except AttributeError:
            pass
        preds_df = pd.DataFrame({"predict": preds})
        if proba is not None:
            for i, class_label in enumerate(model.classes_):
                preds_df[f"p_{class_label}"] = proba[:, i]
        return preds_df
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        raise

def main():
    st.set_page_config(page_title="Loan Eligibility", layout="wide")
    st.title(APP_TITLE)
    st.caption("Enter features or upload a CSV to get predictions. Preprocessing matches training pipeline.")

    try:
        model = load_model()
    except Exception:
        return

    # Batch scoring
    st.sidebar.header("📦 Batch scoring")
    batch_file = st.sidebar.file_uploader("Upload CSV with same features as training (no target).", type=["csv"])
    if batch_file:
        try:
            batch_df = pd.read_csv(batch_file)
            st.sidebar.write("Preview:", batch_df.head())
            if st.sidebar.button("Run batch predictions"):
                preds = predict_df(model, batch_df)
                out = pd.concat([batch_df.reset_index(drop=True), preds], axis=1)
                st.write("Batch predictions:", out.head(20))
                st.download_button("Download predictions.csv", data=out.to_csv(index=False), file_name="predictions.csv")
        except Exception as e:
            st.sidebar.error(f"Batch processing error: {str(e)}")

    st.divider()
    st.subheader("🧮 Single prediction")

    # UI inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self_Employed", ["Yes", "No"])
    property_area = st.selectbox("Property_Area", ["Urban", "Semiurban", "Rural"])
    applicant_income = st.number_input("ApplicantIncome", min_value=0, value=5000, step=100)
    coapplicant_income = st.number_input("CoapplicantIncome", min_value=0, value=0, step=100)
    loan_amount = st.number_input("LoanAmount (in thousands)", min_value=0, value=128, step=1)
    loan_amount_term = st.number_input("Loan_Amount_Term (in days)", min_value=12, value=360, step=12)
    credit_history = st.selectbox("Credit_History", [1.0, 0.0])

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
        try:
            preds = predict_df(model, input_df)
            st.metric("Prediction", preds.iloc[0]["predict"])
            st.write("Raw prediction output:", preds)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

    with st.expander("Show input row as DataFrame"):
        st.dataframe(input_df)

    st.info("Preprocessing matches model training. Ensure CSV uploads use same feature names.")

if __name__ == "__main__":
    main()
