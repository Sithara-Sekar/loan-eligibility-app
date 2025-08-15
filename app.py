#!/usr/bin/env python3
"""
Streamlit app to serve a pre-trained scikit-learn model for Loan Eligibility prediction.
- Automatically detects categorical options from the trained model.
- Supports batch CSV and single-row input.
"""

import os
import pandas as pd
import streamlit as st
import joblib
import numpy as np

APP_TITLE = "🏦 Loan Eligibility (Auto-Categorical)"
MODEL_PATH = os.path.join("models", "leader_model.pkl")

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Could not find '{MODEL_PATH}'. Train the model first.")
    model = joblib.load(MODEL_PATH)
    return model

@st.cache_resource(show_spinner=False)
def get_model_columns():
    """Return the model's feature names"""
    model = load_model()
    return list(model.feature_names_in_)

@st.cache_resource(show_spinner=False)
def get_categorical_options():
    """Return dict: categorical column -> list of categories"""
    model = load_model()
    cat_cols = {}
    # Try to detect categories if model has a preprocessor
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
        if hasattr(preprocessor, "transformers_"):
            for name, transformer, columns in preprocessor.transformers_:
                if transformer.__class__.__name__ == "OneHotEncoder":
                    categories = transformer.categories_
                    for col, cats in zip(columns, categories):
                        cat_cols[col] = list(cats)
    # Fallback: just return generic placeholder options
    return cat_cols

def preprocess_input(df: pd.DataFrame):
    df = df.copy()
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Dependents"]
    cat_cols = [col for col in df.columns if col not in numeric_cols]

    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # Fill missing columns to match model
    model_columns = get_model_columns()
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
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
    st.caption("Enter features or upload a CSV to get predictions. Input fields match the trained model automatically.")

    model = load_model()
    model_columns = get_model_columns()
    cat_options = get_categorical_options()

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

    # Build dynamic input form
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
        # Skip one-hot encoded dummy columns
        if "_" in col and col_
