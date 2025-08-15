# Loan Eligibility — H2O + Streamlit (Hugging Face Space)

This repo trains an H2O AutoML model on the classic Loan Prediction dataset and serves it via a Streamlit app.

## 📦 Repo layout

```
.
├── app.py               # Streamlit app
├── train_h2o.py         # Training script (H2O AutoML)
├── requirements.txt
├── Dockerfile           # For Hugging Face Space (Docker SDK)
└── models/              # Saved model artifacts (created by training script)
    ├── best_model_path.txt
    ├── ...              # H2O model directory/file(s)
    └── leaderboard.csv
```

> **Note**: Run the training script first to populate the `models/` directory before deploying, or include the `models/` folder in your repo. The Streamlit app reads `models/best_model_path.txt` to load the model.

## 🧪 Train locally

1. Ensure you have Java 11+ installed (H2O requires a JVM). If unsure, the Dockerfile in this repo installs OpenJDK 17.
2. Install Python deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Put your `train.csv` in the project root (same directory as `train_h2o.py`).
4. Run AutoML training:
   ```bash
   python train_h2o.py --train_csv train.csv --out_dir models --target Loan_Status --max_runtime_secs 180
   ```
5. This will write:
   - `models/best_model_path.txt` (pointer to the saved model)
   - `models/leaderboard.csv`
   - A saved H2O model directory/file

## ▶️ Run the app locally

```bash
streamlit run app.py
```

Open the printed URL in your browser. You can enter values in the UI or upload a CSV for batch scoring.
Your CSV should contain the following feature columns (no target):
```
Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area
```

## 🚀 Deploy to Hugging Face Spaces (Docker SDK)

Because H2O needs Java, use the **Docker** SDK (not the basic Streamlit SDK).

1. Create a new Space at **hf.co/spaces** → *Create new Space*.
2. Choose **SDK: Docker** (Private or Public).
3. Clone your new Space locally:
   ```bash
   git lfs install
   git clone https://huggingface.co/spaces/<YOUR_USERNAME>/<YOUR_SPACE_NAME>
   cd <YOUR_SPACE_NAME>
   ```
4. Copy the files from this repo into the Space directory:
   ```bash
   cp /path/to/your/repo/{app.py,train_h2o.py,requirements.txt,Dockerfile} .
   # If you already trained locally and want to ship the model:
   cp -r /path/to/your/repo/models ./models
   ```
   > Alternatively, you can push the code first, then start a Space **“Build & Run”**, and train in a local environment—just make sure the `models/` directory ends up included in the Space.
5. Commit & push:
   ```bash
   git add .
   git commit -m "Initial commit: H2O + Streamlit Loan Eligibility"
   git push
   ```
6. The Space will build with your Dockerfile and then start Streamlit automatically on `$PORT` (set to 7860).

### Updating the app or model
- Update code locally, re-run `train_h2o.py`, commit the updated `models/` directory, and push.
- Alternatively, mount a volume and train locally then copy artifacts into the Space repo.

## 🛠️ Troubleshooting

- **Java errors / H2O fails to start**: Ensure the Docker image includes a JRE (the provided Dockerfile installs OpenJDK 17).
- **Model file not found**: Run `train_h2o.py` to generate `models/best_model_path.txt`, or make sure `models/` is present in the Space.
- **Port issues**: Spaces inject `$PORT`; the Dockerfile calls `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
- **Schema mismatch**: If your dataset differs from the classic columns, edit the UI section in `app.py` or use batch CSV scoring.

## 📜 License

MIT (or your preferred license).
