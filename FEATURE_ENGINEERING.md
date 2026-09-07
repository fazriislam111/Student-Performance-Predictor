# Feature Engineering Documentation

## Dataset
Source file: `data/student_data.csv` (708 students, no missing values after cleaning).

Raw columns were renamed for consistency:
| Original column | Renamed to |
|---|---|
| `quiz1_score out of 15` | `quiz1_score` |
| `quiz2_score out of 15` | `quiz2_score` |
| `assignment_score out of 10` | `assignment_score` |
| `total score` | `total_score` |

## Features used by every model
- `Attendance_Rate` (float, %)
- `quiz1_score` (int, 0–15)
- `quiz2_score` (int, 0–15)
- `assignment_score` (int, 0–10)

## Targets
- **Regression target:** `total_score` (continuous)
- **Classification target:** `performance` — categorical with 4 classes, label-encoded in a fixed order: `fail=0, poor=1, average=2, good=3`.

## Train/test split
An 80/20 split (`random_state=42`) was used, **stratified on the classification target** so all four performance classes are proportionally represented in both sets. The same split (same row indices) is reused for both the regression and classification pipelines so results are directly comparable.

## Per-model feature engineering

### Regression pipeline
| Model | Feature engineering |
|---|---|
| Linear Regression | `StandardScaler` fit on training features (zero mean, unit variance) — required because raw features are on very different scales (e.g., 0–100 vs 0–15). |
| Polynomial Regression | `StandardScaler` **then** `PolynomialFeatures(degree=2, include_bias=False)`, generating all quadratic terms and pairwise interactions before fitting a `LinearRegression`. |
| XGBoost | No scaling needed — tree-based models split on raw feature values and are invariant to monotonic transformations. Raw features are used directly. |

### Classification pipeline
| Model | Feature engineering |
|---|---|
| Logistic Regression | `StandardScaler` — required since it is a linear, gradient-based model sensitive to feature scale. |
| Decision Tree | None — raw features (tree splits are scale-invariant). |
| Random Forest | None — raw features (ensemble of trees, scale-invariant). |
| KNN | `StandardScaler` — **essential** for distance-based models; without scaling, `Attendance_Rate` (range ~50–100) would dominate the Euclidean distance over quiz/assignment scores (range 0–15). |
| XGBoost | None — raw features (tree-based, scale-invariant). |

## Label encoding
A single `LabelEncoder` was fit on the fixed class order `[fail, poor, average, good]` and shared across all five classification models and their saved pickles, ensuring predictions decode consistently across the web app.

## Artifacts saved per model
Each pickle file (`models/*.pkl`) bundles everything needed to run inference at serve time:
- The trained `model` object
- The fitted `scaler` (if applicable to that model)
- The fitted `poly` transformer (Polynomial Regression only)
- The fitted `label_encoder` (classification models only)
- The exact `features` list/order expected as input
