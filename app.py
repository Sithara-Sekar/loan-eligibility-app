#!/usr/bin/env python3
"""
Streamlit app to serve the H2O model for Loan Eligibility prediction.
- Loads the best model from leader_model.pkl (written by train_h2o.py).
- Lets users enter features manually or upload a CSV for batch scoring.
"""
import os
import pandas as pd
import streamlit as st
import h2o
import pickle

APP_TITLE = "🏦 Loan Eligibility (H2O + Streamlit)"
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
PICKLE_MODEL_PATH = os.path.join(MODEL_DIR, "leader_model.pkl")

@st.cache_resource(show_spinner=False)
def init_h2o(max_mem="2G"):
    # Bind to localhost; Spaces will only expose Streamlit externally
    h2o.init(nthreads=-1, max_mem_size=max_mem, ip="127.0.0.1", port=54321)
    return True

@st.cache_resource(show_spinner=False)
def load_model_from_pickle():
    if not os.path.exists(PICKLE_MODEL_PATH):
        raise FileNotFoundError(f"Could not find '{PICKLE_MODEL_PATH}'. Train the model first.")
    with open(PICKLE_MODEL_PATH, "rb") as f:
        model_info = pickle.load(f)
    model_path = model_info["model_path"]
    model = h2o.load_model(model_path)
    return model

def predict_df(model, df: pd.DataFrame):
    # Convert pandas DataFrame to H2OFrame for prediction
    hf = h2o.H2OFrame(df)
    preds = model.predict(hf).as_data_frame()
    return preds

def main():
    st.set_page_config(page_title="Loan Eligibility (H2O)", layout="wide")
    st.title(APP_TITLE)
    st.caption("Powered by H2O AutoML. Enter features below or upload a CSV to get predictions.")

    # Start H2O & load the model from pickle
    init_h2o(max_mem=os.environ.get("H2O_MAX_MEM", "2G"))
    model = load_model_from_pickle()

    # Sidebar: batch scoring
    st.sidebar.header("📦 Batch scoring")
    batch_file = st.sidebar.file_uploader("Upload CSV with the same feature columns as training data (excluding target).", type=["csv"])
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

    # Build simple UI using known columns
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

    st.info("Tip: If you trained with a different schema, make sure to update the UI fields or upload a CSV for batch scoring.")

if __name__ == "__main__":
    main()
