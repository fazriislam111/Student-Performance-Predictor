# Student Performance Prediction — End-to-End ML Project

A complete machine learning + web application project that predicts a student's
**total score** (regression) and **performance category** (classification: fail / poor /
average / good) from their attendance rate, two quiz scores, and an assignment score.

Retrained (September 2026) on an updated 2,000-row dataset — see `FEATURE_ENGINEERING.md`
for the exact column mapping and how to retrain again on a future dataset.

## Project Structure
```
project/
├── app.py                     # Flask web application (routes + prediction API)
├── train.py                   # Full training pipeline (EDA + 8 models + all plots)
├── requirements.txt           # Python dependencies
├── Procfile / render.yaml / runtime.txt / .gitignore   # Deployment config (Render)
├── FEATURE_ENGINEERING.md     # Documentation of preprocessing per model
├── data/
│   ├── student_data_raw.csv   # Original uploaded dataset
│   └── student_data.csv       # Cleaned/renamed copy (written by train.py)
├── models/                    # Pickled trained models (8) + metadata/results JSON
│                               # (XGBoost models saved as native JSON, see below)
├── plots/                     # Master copy of all generated PNGs (eda/analysis/importance)
├── static/
│   ├── style.css
│   └── plots/                 # Copies served by Flask (eda/analysis/importance)
└── templates/                 # Jinja2 HTML templates (index, eda, predict, analyze)
```

## Part 1 — Machine Learning Pipeline (`train.py`)

Run once to regenerate everything (models + plots):
```bash
pip install -r requirements.txt
python3 train.py
```

This will:
1. Load `data/student_data_raw.csv`, rename columns to internal names, drop any
   missing rows, and write the cleaned copy to `data/student_data.csv`.
2. Generate **8 EDA plots** (distributions, correlation heatmap, class imbalance,
   feature relationships, boxplots by class, pairplot, missing-values check) into `plots/eda/`.
3. Train **3 regression models** (Linear Regression, XGBoost, Polynomial Regression)
   to predict `total_score`.
4. Train **5 classification models** (Logistic Regression, Decision Tree, Random
   Forest, KNN, XGBoost) to predict `performance`.
5. For every model, generate interpretation plots into `plots/analysis/`
   (residuals, decision boundaries, tree structure, feature importance,
   OOB error curves, confusion matrices, metric bar charts, and a
   training/validation curve for every model — a boosting loss curve for the
   two XGBoost models, a train-size learning curve for the other six).
6. Generate a **top-3 feature importance** chart per model into `plots/importance/`.
7. Save all 8 trained models into `models/` (standard pickle bundles for 6 models;
   XGBoost's 2 models save via native JSON + a small pointer pickle — see
   `FEATURE_ENGINEERING.md` for why).
8. Save `models/results.json` and `models/metadata.json`, which `app.py` reads
   directly — retraining and restarting the app is all that's needed to update the UI.

## Part 2 — Web Interface (`app.py`)

| Page | Route | Purpose |
|---|---|---|
| EDA | `/eda` | Dataset shape/column dtypes, a 5-row random sample, and all EDA plots with explanations |
| Predict Performance | `/predict` | Choose Regression/Classification → pick a model → enter feature values → live prediction |
| Analyze Models | `/analyze` | Choose a model → hyperparameters, feature engineering, top-3 features, all interpretation plots |

### Run locally
```bash
pip install -r requirements.txt
python3 app.py
```
Open **http://127.0.0.1:5001** (port 5001 avoids macOS's AirPlay Receiver conflict on port 5000).

### Deploy (Render — free tier)
This repo already has `Procfile`, `render.yaml`, and `runtime.txt`. On [render.com](https://render.com):
**New → Web Service** → connect your GitHub repo → confirm build command
`pip install -r requirements.txt` and start command `gunicorn app:app --bind 0.0.0.0:$PORT`
→ Free plan → Deploy. First build takes a few minutes (xgboost/shap/matplotlib are heavy).

Netlify **cannot** run this app — it only hosts static sites/serverless functions, not a
persistent Flask/WSGI server with these ML dependencies.

## Notes
- With the new 2,000-row dataset, model performance improved substantially over the
  original 708-row dataset: regression R² ≈ 0.81–0.83, classification accuracy ≈ 80–82%.
- The "poor" performance class is rare (25 of 2,000 rows) — expect lower recall on that
  class specifically; this is a property of the data's class balance, visible in the
  class-imbalance EDA plot and each model's confusion matrix.
- Decision-boundary plots (Logistic Regression, Decision Tree, KNN) use only the two
  most informative features for 2D visualization; the deployed prediction models still
  use all four features.
