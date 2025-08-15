#!/usr/bin/env python3
"""
Train an H2O model for Loan Eligibility and save it for deployment.
- Expects train.csv and test.csv in the working directory (same columns as Analytics Vidhya/Kaggle Loan Prediction dataset).
- Uses H2O AutoML to find a strong model.
- Saves the best model and records its path for the Streamlit app to load.
"""

import os
import argparse
import pandas as pd
import h2o
from h2o.automl import H2OAutoML

def main(args):
    # Init H2O (ensure Java is available; handled by Dockerfile/requirements on Spaces)
    h2o.init(nthreads=-1, max_mem_size=args.max_mem_size, ip="127.0.0.1", port=54321)
    
    # Paths
    train_csv = "train.csv" 
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load train CSV with pandas only to log shapes; H2O will read via H2OFrame for training
    pdf = pd.read_csv(train_csv)
    print(f"Loaded train.csv with shape: {pdf.shape} and columns: {list(pdf.columns)}")

    # Convert to H2OFrame
    hf = h2o.H2OFrame(pdf)
    
    # Target and features
    target = args.target
    if target not in hf.columns:
        raise ValueError(f"Target column '{target}' not found in columns: {hf.columns}")

    # Make sure target is categorical
    hf[target] = hf[target].asfactor()

    # Simple split for validation (you can also provide a separate valid dataset if desired)
    train, valid = hf.split_frame(ratios=[0.8], seed=42)

    # Set up and run AutoML
    aml = H2OAutoML(
        max_runtime_secs=args.max_runtime_secs,
        seed=42,
        sort_metric="AUC",
        stopping_metric="AUC",
        project_name="loan_eligibility_automl",
        exclude_algos=None,  # or e.g., ["DeepLearning"] if you want to restrict
        nfolds=0  # using simple train/valid split
    )
    aml.train(y=target, training_frame=train, validation_frame=valid)

    # Leaderboard and best model
    lb = aml.leaderboard.as_data_frame()
    print("AutoML Leaderboard (top 10):")
    print(lb.head(10))

    leader = aml.leader
    print("Best model:", leader.model_id)

    # Save model (native H2O model)
    saved_model_path = h2o.save_model(model=leader, path=out_dir, force=True)
    print("Saved best model to:", saved_model_path)

    # Also save MOJO for Java-only scoring if you ever need it
    try:
        mojo_path = leader.download_mojo(path=out_dir, get_genmodel_jar=False)
        print("Saved MOJO to:", mojo_path)
    except Exception as e:
        print("Could not save MOJO:", e)

    # Persist a pointer file so the Streamlit app can find the model
    with open(os.path.join(out_dir, "best_model_path.txt"), "w") as f:
        f.write(saved_model_path)

    # Save leaderboard for reference
    lb.to_csv(os.path.join(out_dir, "leaderboard.csv"), index=False)

    # Shutdown H2O (optional in scripts)
    h2o.shutdown(prompt=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="train.csv", help="Path to training CSV")
    parser.add_argument("--out_dir", type=str, default="models", help="Where to save models/outputs")
    parser.add_argument("--target", type=str, default="Loan_Status", help="Target column name")
    parser.add_argument("--max_runtime_secs", type=int, default=120, help="AutoML max runtime seconds")
    parser.add_argument("--max_mem_size", type=str, default="2G", help="Max memory size for H2O cluster, e.g. '2G'")
    args = parser.parse_args()
    main(args)
