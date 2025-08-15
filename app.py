#!/usr/bin/env python3
"""
Streamlit app to serve a pre-trained scikit-learn model for Loan Eligibility prediction.
- Works with models/leader_model.pkl
- Dynamically matches input to the model's expected features
"""

import os
import pandas as pd
import streamlit as st
import joblib

APP_TITLE = "🏦 Loan Eligibility (Fixed Version)"
MODEL_PATH = os.path.join("models", "leader_model.pkl")

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Could not find '{MODEL_PATH}'. Train the model first.")
    model = joblib.load(MODEL_PATH)
    return model

@st.cache_resource(show_spinner=False)
def get_model_columns():
    model = load_model()
    return list(model.feature_names_in_)

def preprocess_input(df: pd.DataFrame):
    """Preprocess input to match training schema (one-hot encoding + missing columns)."""
    df = df.copy()

    # Map '3+' to 3 in Dependents
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

    # Identify numeric columns (assumes common numeric features)
    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Dependents"]
    cat_cols = [col for col in df.columns if col not in numeric_cols]

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # Fill missing columns with zeros to match model
    model_columns = get_model_columns()
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match model
    df = df[model_columns]
    return df

def predict_df(model, df: pd.DataFrame):
    df_processed = preprocess_input(df)
    preds = model.predict(df_processed)
    preds_df = pd.DataFrame({"predict": preds})
    try:
        proba = model.predict_proba(df_processed)
        for i, class_label in enumerate(model.classes_):
            preds_df[f"p_{class_label}"] = proba[:, i]
    except AttributeError:
        pass
    return preds_df

def main():
    st.set_page_config(page_title="Loan Eligibility", layout="wide")
    st.title(APP_TITLE)
    st.caption("Enter features or upload a CSV to get predictions. Input is matched automatically to the model's features.")

    model = load_model()
    model_columns = get_model_columns()

    # Sidebar: batch scoring
    st.sidebar.header("📦 Batch scoring")
    batch_file = st.sidebar.file_uploader("Upload CSV (exclude target)", type=["csv"])
    if batch_file:
        batch_df = pd.read_csv(batch_file)
        st.sidebar.write("Preview:", batch_df.head())
        if st.sidebar.button("Run batch predictions"):
            preds = predict_df(model, batch_df)
            out = pd.concat([batch_df.reset_index(drop=True), preds], axis=1)
            st.write("Batch predictions:", out.head(20))
            st.download_button("Download predictions.csv", data=out.to_csv(index=False), file_name="predictions.csv")

    st.divider()
    st.subheader("🧮 Single prediction")

    # Dynamically generate UI based on model columns
    input_data = {}
    numeric_defaults = {
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 0,
        "LoanAmount": 128,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Dependents": "0"
    }

    for col in model_columns:
        # Skip one-hot dummy columns (they will be created automatically)
        if "_" in col and col not in numeric_defaults:
            continue
        if col in numeric_defaults:
            input_data[col] = st.number_input(col, min_value=0, value=numeric_defaults[col], step=100)
        else:
            # Provide a selectbox for categorical columns
            # If you know the original categories, replace with correct list
            input_data[col] = st.selectbox(col, ["Option1", "Option2", "Option3"], index=0)

    input_df = pd.DataFrame([input_data])

    if st.button("Predict eligibility"):
        preds = predict_df(model, input_df)
        st.metric("Prediction", preds.iloc[0]["predict"])
        st.write("Raw prediction output:", preds)

    with st.expander("Show input row as DataFrame"):
        st.dataframe(input_df)

    st.info("Preprocessing ensures your input always matches the model's expected features.")

if __name__ == "__main__":
    main()
