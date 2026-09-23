# Feature Engineering Documentation

## Dataset
Source file: `data/student_data_raw.csv` (2,000 students, no missing values).
A cleaned, renamed copy is written to `data/student_data.csv` by `train.py` on every run — this is the file the Flask app reads for the EDA page's dataset overview.

Raw columns were renamed to internal names used everywhere downstream (unchanged from the original project, so `app.py` needed no changes when this dataset was swapped in):

| Raw column | Renamed to |
|---|---|
| `student_ID` | `Student_ID` |
| `attendance (%)` | `Attendance_Rate` |
| `quiz_1 score (out of 15)` | `quiz1_score` |
| `quiz_2 score (out of 15)` | `quiz2_score` |
| `assignment score (out of 10)` | `assignment_score` |
| `final score (out of 100)` | `total_score` |

## Features used by every model
- `Attendance_Rate` (int, 52–100%)
- `quiz1_score` (int, 6–15)
- `quiz2_score` (int, 6–15)
- `assignment_score` (int, 4–10)

## Targets
- **Regression target:** `total_score` (continuous, 25–100)
- **Classification target:** `performance` — 4 classes, label-encoded in a fixed order: `fail=0, poor=1, average=2, good=3`. Class counts: average 1168, fail 595, good 212, poor 25 (this dataset is noticeably imbalanced toward "average").

## Train/test split
An 80/20 split (`random_state=42`), **stratified on the classification target**, reused across both pipelines for directly comparable results.

## Per-model feature engineering
Identical approach to the original dataset — only the underlying numbers changed:

| Model | Feature engineering |
|---|---|
| Linear Regression | `StandardScaler` on all 4 features. |
| Polynomial Regression | `StandardScaler` then `PolynomialFeatures(degree=2, include_bias=False)`. |
| XGBoost (regression & classification) | None — raw features, scale-invariant. |
| Logistic Regression | `StandardScaler`. |
| Decision Tree | None. |
| Random Forest | None. |
| KNN | `StandardScaler` — essential for this distance-based model. |

## Label encoding
A single `LabelEncoder` fit on `[fail, poor, average, good]`, shared across all classification models and pickles.

## Artifacts saved per model
Each `models/*.pkl` bundles what's needed at inference time: the trained `model` (or, for XGBoost, a pointer to its native JSON file — see note below), the fitted `scaler` if applicable, the fitted `poly` transformer (Polynomial Regression only), the fitted `label_encoder` (classification only), and the `features` list/order.

**XGBoost note:** `regression_xgboost.pkl` and `classification_xgboost.pkl` store `{"model_type", "native_path", "features"}` rather than the model object directly — the actual model is saved separately as `regression_xgboost_native.json` / `classification_xgboost_native.json` via XGBoost's own `save_model()`. This avoids pickle-format incompatibilities across XGBoost versions. `app.py`'s `load_bundle()` detects this format and reconstructs the model automatically; no manual steps needed.

## Retraining on a new dataset
To swap in another dataset: drop the new CSV at `data/student_data_raw.csv`, update the column-rename dict near the top of `train.py` if the raw column names differ, and run `python3 train.py`. Every downstream file name, model name, and plot name stays identical, so `app.py` and the templates require zero changes.
