# Student Performance Prediction — End-to-End ML Project

A complete machine learning + web application project that predicts a student's
**total score** (regression) and **performance category** (classification: fail / poor /
average / good) from their attendance rate, two quiz scores, and an assignment score.

## Project Structure
```
project/
├── app.py                     # Flask web application (routes + prediction API)
├── train.py                   # Full training pipeline (EDA + 8 models + all plots)
├── requirements.txt           # Python dependencies
├── FEATURE_ENGINEERING.md     # Documentation of preprocessing per model
├── data/
│   └── student_data.csv       # Source dataset
├── models/                    # Pickled trained models (8) + metadata/results JSON
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
1. Load and clean `data/student_data.csv`.
2. Generate **7 EDA plots** (distributions, correlation heatmap, class imbalance,
   feature relationships, boxplots by class, pairplot) into `plots/eda/`.
3. Train **3 regression models** (Linear Regression, XGBoost, Polynomial Regression)
   to predict `total_score`.
4. Train **5 classification models** (Logistic Regression, Decision Tree, Random
   Forest, KNN, XGBoost) to predict `performance`.
5. For every model, generate interpretation plots into `plots/analysis/`
   (residuals, decision boundaries, tree structure, SHAP/gain feature importance,
   OOB error curves, loss curves, confusion matrices, metric bar charts — tailored
   per model type as requested).
6. Generate a **top-3 feature importance** chart per model into `plots/importance/`.
7. Save all 8 trained models as pickle files in `models/`, bundled together with
   any scaler / polynomial transformer / label encoder they need at inference time.
8. Save `models/results.json` (metrics, hyperparameters, top-3 features per model)
   and `models/metadata.json` (feature list, ranges, class order) — consumed
   directly by the Flask app so the UI always reflects the latest training run.

See `FEATURE_ENGINEERING.md` for exactly what preprocessing was applied to each model.

## Part 2 — Web Interface (`app.py`)

A single Flask app with three pages, as specified:

| Page | Route | Purpose |
|---|---|---|
| EDA | `/eda` | All EDA plots with title + short explanation each |
| Predict Performance | `/predict` | Choose Regression/Classification → pick a model → enter feature values → get a live prediction (calls `/api/predict`) |
| Analyze Models | `/analyze` | Choose a model → see its hyperparameters, feature engineering notes, top-3 features, and all interpretation plots (calls `/api/analyze/<model>`) |

### Run the web app
```bash
pip install -r requirements.txt
python3 app.py
```
Then open **http://127.0.0.1:5000** in your browser.

The app loads pickled models **lazily and caches them in memory**, and reads
`models/metadata.json` / `models/results.json` for everything else — so if you
retrain (`python3 train.py`) and restart the Flask app, the UI updates automatically.

## Notes
- The dataset is small (708 rows) and the four available features only weakly
  determine total score/performance (test R² and accuracy are modest for all
  models) — this is a property of the input data, not a training bug. The
  metrics and plots are generated faithfully as an honest performance analysis.
- Decision-boundary plots (Logistic Regression, Decision Tree, KNN) are trained
  on the two most informative features only (`Attendance_Rate`, `quiz1_score`)
  purely for 2D visualization — the deployed prediction models still use all
  four features.
