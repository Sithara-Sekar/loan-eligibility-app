# train_h2o.py
from pathlib import Path
import pandas as pd
import h2o
from h2o.automl import H2OAutoML

# -------- Paths (robust to where you run from) --------
HERE = Path(__file__).parent
TRAIN_CSV = HERE / "train.csv"          # adjust if your CSV is elsewhere
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET = "Loan_Status"                  # <-- change if your target has a different name

def main():
    print(f"[info] Script folder: {HERE}")
    print(f"[info] Reading CSV: {TRAIN_CSV}")
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"CSV not found at: {TRAIN_CSV}")

    pdf = pd.read_csv(TRAIN_CSV)
    print(f"[info] Data shape: {pdf.shape}")
    if TARGET not in pdf.columns:
        raise ValueError(f"Target '{TARGET}' not in columns: {list(pdf.columns)}")

    # Start H2O
    h2o.init()

    # To H2OFrame
    hf = h2o.H2OFrame(pdf)

    # Make target categorical for classification (comment this if doing regression)
    hf[TARGET] = hf[TARGET].asfactor()

    features = [c for c in hf.columns if c != TARGET]
    print(f"[info] Using {len(features)} features.")

    # Basic sanity: drop rows with all-NA in features
    # (H2O usually handles NAs, but if everything is NA it can break)
    # Optional: do hf = hf.dropna() if your CSV is very dirty.

    # Split
    train, test = hf.split_frame(ratios=[0.8], seed=1234)

    # AutoML
    aml = H2OAutoML(max_runtime_secs=120, seed=42)
    aml.train(x=features, y=TARGET, training_frame=train)

    lb = aml.leaderboard
    print("[info] Leaderboard (top 5):")
    print(lb.head(rows=5))

    if aml.leader is None:
        raise RuntimeError("AutoML did not produce a leader model. Check target/type/rows.")

    # Save model next to this script, under ./models
    saved_path = h2o.save_model(model=aml.leader, path=str(MODELS_DIR), force=True)
    print(f"[success] Saved model to: {saved_path}")

    # Optional: quick metric
    perf = aml.leader.model_performance(test)
    try:
        print("[info] AUC:", perf.auc())
    except Exception:
        pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[error]", e)
        raise
