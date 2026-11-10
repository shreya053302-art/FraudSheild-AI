# 🛡️ FraudShield AI

An AI-powered financial fraud detection app built with **Streamlit + scikit-learn**.
It trains a real model on the credit card fraud dataset, scores single transactions or whole CSV files,
explains every flag, stores cases in SQLite and generates PDF/CSV reports.

> **Decision-support tool.** Predictions are statistical estimates, not proof of fraud.

## What it does

| Page | What works |
|---|---|
| Dashboard | KPIs, trend, pie, hourly activity, risk distribution, recent suspicious table, alerts (all from saved data) |
| Analyze Transaction | Form with validation → trained model → risk score 0-100, level, confidence, explanation, recommended actions |
| Bulk Analysis | CSV upload, validation, preview, scoring, real precision/recall if the file has `Class`, download analyzed CSV |
| Investigation Center | Search + filters, status, **Mark as Reviewed**, notes, CSV and PDF export |
| Analytics & Reports | Daily/weekly/monthly trends, amount analysis, risk heatmap, fraud patterns, model metrics, confusion matrix, PR curve, feature importance, monitoring, PDF + CSV reports |
| ShieldBot | Chat assistant. Built-in answers by default; optional Gemini mode through an environment variable |
| Model Center | Model status and a training screen (use `creditcard.csv` or upload any CSV with a 0/1 target) |
| Settings | Preferences, system health, delete data |

## 1. Install (Windows, VS Code)

Open the project folder in VS Code, then open **Terminal → New Terminal**:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` once and activate again.
(Mac/Linux: `source .venv/bin/activate`.)

## 2. Get the dataset

1. Create a free account at Kaggle and open **Credit Card Fraud Detection** (dataset by *mlg-ulb*):
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click **Download** and unzip it. You get `creditcard.csv` (284,807 rows, 31 columns, target `Class`).
3. Put the file at **`data/creditcard.csv`**.

No dataset yet? You can still test everything with **synthetic** data:

```powershell
python scripts/make_sample_data.py
```

That creates `data/sample_synthetic_transactions.csv`. It is fake data in the same layout. Use it only to check the app works;
the metrics it gives mean nothing about real fraud.

## 3. Run

```powershell
streamlit run app.py
```

The browser opens at http://localhost:8501.

1. Go to **🧠 Model Center** → pick `data/creditcard.csv` → **Load dataset** → **Train model**
   (use "Max rows = 50000" for a quick first try; the full dataset takes a few minutes).
2. Go to **📂 Bulk Analysis** and upload a CSV, or **🔍 Analyze Transaction** to score one transaction.
3. Review flagged cases in **🕵️ Investigation Center**.

Your model (`models/fraud_model.joblib`) and data (`data/fraudshield.db`) are saved on disk, so everything is still
there after a restart.

## 4. ShieldBot AI mode (optional, free)

1. Get a key at https://aistudio.google.com/apikey
2. Copy `.env.example` to `.env` and paste the key after `GEMINI_API_KEY=`
3. Restart the app.

The key stays on the server in an environment variable. It is never stored in the database or sent to the browser.
Without a key, ShieldBot answers from its built-in knowledge base.
(The Gemini connection could not be tested while building this project; the built-in mode is fully tested.)

## 5. Run the tests

```powershell
python -m unittest discover -s tests -v
```

Tests use a temporary database and synthetic data, so your real files are untouched.

## How the ML works (simple version)

1. **Load** the CSV and find the target column (`Class`).
2. **Split first**: 80% train / 20% test, then train is split again into fit and validation parts (all stratified).
3. **Preprocess inside a pipeline**: median imputer + RobustScaler on `Time` and `Amount`, fitted on training rows only.
4. **Imbalance**: fraud is only ~0.17% of rows, so both models use **class weights**. We do not oversample (e.g. SMOTE), so
   no artificial rows can leak into validation/test data.
5. **Train** Random Forest and Logistic Regression.
6. **Choose the threshold** on the validation set (best F1), and **choose the model** by validation PR-AUC.
7. **Report honest numbers** on the untouched test set: precision, recall, F1, PR-AUC, confusion matrix. Accuracy is shown
   too, but it is misleading when 99.8% of rows are legitimate.
8. **Save** with Joblib; the app loads that file for every prediction.

**Risk score** = the model's raw fraud score rescaled to 0-100 so the decision threshold sits at 50
(Low <25, Medium 25-50, High 50-75, Critical ≥75; 50+ is flagged).
**Confidence** = how far the score is from that decision line. It is *not* a calibrated probability.
**Explanations** = the features that are far from the typical legitimate value (z-score) and that the model relies on.
They explain what looks unusual; they are not a proof of cause (SHAP is not used).

## Project structure

```
app.py                     entry point, navigation
backend/
  config.py                paths, limits, environment variables
  database.py              SQLite tables and functions
  preprocessing.py         CSV reading/validation, target detection, pipeline
  fraud_detection.py       training, evaluation, model saving
  prediction.py            loading model, risk score, explanations, run_analysis()
  analytics.py             dashboard and analytics numbers from the database
  report_generator.py      PDF and CSV reports (ReportLab)
  assistant.py             ShieldBot (Gemini optional + built-in answers)
frontend/
  styles.py                dark theme CSS and UI components
  common.py                page list and helpers
  dashboard.py, detection.py, bulk_analysis.py, investigation.py,
  analytics_page.py, assistant.py, model_center.py, settings.py
scripts/make_sample_data.py   synthetic test data (NOT real)
tests/test_app.py             pipeline + page tests
```

## Security notes

- Uploads: `.csv` only, size and row limits, parsed by pandas (never executed), numeric validation.
- All SQL uses parameters. No passwords exist (single-user local app). API keys come from environment variables only.
- Only load model files you trained yourself (Joblib files can run code when loaded).

## Known limits

- No login or multi-user support (single local user).
- "Real-time" alerts are pop-ups and the alerts panel right after an analysis, not a live stream.
- Bulk runs save only the 5,000 riskiest rows (risk ≥ 25) to the database; all rows still count in totals and the download.
- In the Kaggle data, `Time` is seconds since the first transaction, so "hour" charts are relative hours.
- Everything is designed for the credit card dataset layout, but the Model Center can train on other numeric CSVs.
