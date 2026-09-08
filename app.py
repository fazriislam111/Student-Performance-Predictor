import json
import os
import pickle
import random

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "student_data.csv")

RAW_COLUMN_RENAME = {
    "quiz1_score out of 15": "quiz1_score",
    "quiz2_score out of 15": "quiz2_score",
    "assignment_score out of 10": "assignment_score",
    "total score": "total_score",
}


def load_dataframe():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=RAW_COLUMN_RENAME)
    return df

with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    METADATA = json.load(f)
with open(os.path.join(MODEL_DIR, "results.json")) as f:
    RESULTS = json.load(f)

FEATURES = METADATA["features"]
CLASS_ORDER = METADATA["class_order"]

FEATURE_LABELS = {
    "Attendance_Rate": "Attendance Rate (%)",
    "quiz1_score": "Quiz 1 Score (out of 15)",
    "quiz2_score": "Quiz 2 Score (out of 15)",
    "assignment_score": "Assignment Score (out of 10)",
}

REGRESSION_MODELS = {
    "linear_regression": {
        "label": "Linear Regression",
        "pickle": "regression_linear_regression.pkl",
    },
    "xgboost_regression": {
        "label": "XGBoost",
        "pickle": "regression_xgboost.pkl",
    },
    "polynomial_regression": {
        "label": "Polynomial Regression",
        "pickle": "regression_polynomial_regression.pkl",
    },
}

CLASSIFICATION_MODELS = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "pickle": "classification_logistic_regression.pkl",
    },
    "decision_tree": {
        "label": "Decision Tree",
        "pickle": "classification_decision_tree.pkl",
    },
    "random_forest": {
        "label": "Random Forest",
        "pickle": "classification_random_forest.pkl",
    },
    "knn": {
        "label": "K-Nearest Neighbors",
        "pickle": "classification_knn.pkl",
    },
    "xgboost_classification": {
        "label": "XGBoost",
        "pickle": "classification_xgboost.pkl",
    },
}

_model_cache = {}


def load_bundle(pickle_name):
    if pickle_name not in _model_cache:
        with open(os.path.join(MODEL_DIR, pickle_name), "rb") as f:
            bundle = pickle.load(f)

        # XGBoost models are stored via their native save format (version-stable)
        # rather than raw pickle, because pickling a Booster directly breaks
        # across different xgboost versions/platforms. Reconstruct here.
        model_type = bundle.get("model_type")
        if model_type == "xgb_regressor":
            from xgboost import XGBRegressor
            m = XGBRegressor()
            m.load_model(os.path.join(MODEL_DIR, bundle["native_path"]))
            bundle["model"] = m
        elif model_type == "xgb_classifier":
            from xgboost import XGBClassifier
            m = XGBClassifier()
            m.load_model(os.path.join(MODEL_DIR, bundle["native_path"]))
            bundle["model"] = m

        _model_cache[pickle_name] = bundle
    return _model_cache[pickle_name]


# ---------------------------------------------------------------------------
# EDA plot descriptions
# ---------------------------------------------------------------------------
EDA_PLOTS = [
    {
        "file": "01_feature_distributions.png",
        "title": "Feature Distributions",
        "desc": "Histograms of attendance rate, both quiz scores, and the assignment score. Most features are fairly evenly spread, with attendance skewing slightly toward higher percentages, helping check for skew before modeling.",
    },
    {
        "file": "02_total_score_distribution.png",
        "title": "Total Score Distribution",
        "desc": "Shows the spread of the total score target used for regression. The distribution is roughly bell-shaped with a wide range, indicating no extreme outliers dominate the target variable.",
    },
    {
        "file": "03_class_imbalance.png",
        "title": "Performance Class Imbalance",
        "desc": "Counts of students in each performance category (fail, poor, average, good). 'Average' is the largest class while 'good' is the smallest, showing a moderate class imbalance to account for during evaluation.",
    },
    {
        "file": "04_correlation_heatmap.png",
        "title": "Correlation Heatmap",
        "desc": "Pairwise correlations between the four features and total score. Quiz and assignment scores show the strongest positive correlation with total score, guiding which features matter most.",
    },
    {
        "file": "05_feature_relationships.png",
        "title": "Feature Relationships with Total Score",
        "desc": "Scatter plots of each feature against total score, colored by performance category. Clusters of colors along the score axis show how performance tiers relate to underlying feature values.",
    },
    {
        "file": "06_boxplots_by_class.png",
        "title": "Feature Spread by Performance Category",
        "desc": "Boxplots comparing each feature's distribution across the four performance categories. Higher-performing students tend to show higher medians and tighter spread in quiz and assignment scores.",
    },
    {
        "file": "07_pairplot.png",
        "title": "Pairwise Feature Relationships",
        "desc": "A full pairplot of all features colored by performance category, revealing how combinations of features separate (or overlap between) the different performance groups.",
    },
]

# ---------------------------------------------------------------------------
# Per-model analysis configuration (plots, descriptions)
# ---------------------------------------------------------------------------
ANALYSIS_CONFIG = {
    "linear_regression": {
        "kind": "regression",
        "label": "Linear Regression",
        "plots": [
            ("linear_regression_analysis.png", "Residuals, predicted-vs-actual scatter, and standardized coefficient importance."),
            ("linear_regression_metrics.png", "Test-set RMSE, MAE, and R² for this model."),
        ],
    },
    "polynomial_regression": {
        "kind": "regression",
        "label": "Polynomial Regression",
        "plots": [
            ("polynomial_regression_analysis.png", "Polynomial fit visualization against attendance rate plus residual analysis."),
            ("polynomial_regression_metrics.png", "Test-set RMSE, MAE, and R² for this model."),
        ],
    },
    "xgboost_regression": {
        "kind": "regression",
        "label": "XGBoost",
        "plots": [
            ("xgboost_regression_analysis.png", "Training/validation RMSE loss curve, gain-based feature importance, and predicted-vs-actual scatter."),
            ("xgboost_regression_metrics.png", "Test-set RMSE, MAE, and R² for this model."),
        ],
    },
    "logistic_regression": {
        "kind": "classification",
        "label": "Logistic Regression",
        "plots": [
            ("logistic_regression_boundary.png", "Decision boundary visualized on the two most informative features (Attendance Rate vs Quiz 1)."),
            ("logistic_regression_analysis.png", "Per-class coefficient heatmap and predicted-probability distributions."),
            ("logistic_regression_confusion.png", "Confusion matrix on the held-out test set."),
            ("logistic_regression_metrics.png", "Accuracy, precision, recall, and F1 on the test set."),
        ],
    },
    "decision_tree": {
        "kind": "classification",
        "label": "Decision Tree",
        "plots": [
            ("decision_tree_analysis.png", "Visualization of the tree's top splits (first 3 levels) showing how it partitions students."),
            ("decision_tree_boundary.png", "Decision boundary on the two most informative features, showing the tree's rectangular partitions."),
            ("decision_tree_confusion.png", "Confusion matrix on the held-out test set."),
            ("decision_tree_metrics.png", "Accuracy, precision, recall, and F1 on the test set."),
        ],
    },
    "random_forest": {
        "kind": "classification",
        "label": "Random Forest",
        "plots": [
            ("random_forest_analysis.png", "Out-of-bag error rate as trees are added, plus feature importance across the forest."),
            ("random_forest_confusion.png", "Confusion matrix on the held-out test set."),
            ("random_forest_metrics.png", "Accuracy, precision, recall, and F1 on the test set."),
        ],
    },
    "knn": {
        "kind": "classification",
        "label": "K-Nearest Neighbors",
        "plots": [
            ("knn_boundary.png", "Decision boundary (k=9) on the two most informative features."),
            ("knn_analysis.png", "Test accuracy as the number of neighbors (k) changes, showing neighbor influence."),
            ("knn_confusion.png", "Confusion matrix on the held-out test set."),
            ("knn_metrics.png", "Accuracy, precision, recall, and F1 on the test set."),
        ],
    },
    "xgboost_classification": {
        "kind": "classification",
        "label": "XGBoost",
        "plots": [
            ("xgboost_classification_analysis.png", "Training/validation log-loss curve and gain-based feature importance."),
            ("xgboost_classification_confusion.png", "Confusion matrix on the held-out test set."),
            ("xgboost_classification_metrics.png", "Accuracy, precision, recall, and F1 on the test set."),
        ],
    },
}


def clean_params(params):
    """Keep only simple, human-readable, non-null hyperparameters."""
    out = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, float) and (v != v):  # NaN check
            continue
        if isinstance(v, (int, float, str, bool)):
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/eda")
def eda():
    df = load_dataframe()

    shape_info = {"rows": df.shape[0], "columns": df.shape[1]}
    dtypes_info = [{"column": col, "dtype": str(dtype)} for col, dtype in df.dtypes.items()]

    sample_df = df.sample(n=5, random_state=random.randint(0, 1_000_000)).reset_index(drop=True)
    if "Attendance_Rate" in sample_df.columns:
        sample_df["Attendance_Rate"] = sample_df["Attendance_Rate"].round(2)
    sample_columns = list(sample_df.columns)
    sample_rows = sample_df.to_dict(orient="records")

    return render_template(
        "eda.html",
        plots=EDA_PLOTS,
        shape_info=shape_info,
        dtypes_info=dtypes_info,
        sample_columns=sample_columns,
        sample_rows=sample_rows,
    )


@app.route("/predict")
def predict_page():
    return render_template(
        "predict.html",
        reg_models=REGRESSION_MODELS,
        cls_models=CLASSIFICATION_MODELS,
        features=FEATURES,
        feature_labels=FEATURE_LABELS,
        feature_ranges=METADATA["feature_ranges"],
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)
    task = payload.get("task")  # "regression" or "classification"
    model_key = payload.get("model")
    raw_features = payload.get("features", {})

    try:
        input_row = {f: float(raw_features[f]) for f in FEATURES}
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Please provide valid numeric values for all features."}), 400

    X_input = pd.DataFrame([input_row])[FEATURES]

    try:
        if task == "regression":
            if model_key not in REGRESSION_MODELS:
                return jsonify({"error": "Unknown regression model."}), 400
            bundle = load_bundle(REGRESSION_MODELS[model_key]["pickle"])
            model = bundle["model"]
            if "scaler" in bundle and bundle["scaler"] is not None:
                X_proc = bundle["scaler"].transform(X_input)
            else:
                X_proc = X_input.values
            if "poly" in bundle:
                X_proc = bundle["poly"].transform(X_proc)
            pred = float(model.predict(X_proc)[0])
            return jsonify({"prediction": round(pred, 2), "label": "Predicted Total Score"})

        elif task == "classification":
            if model_key not in CLASSIFICATION_MODELS:
                return jsonify({"error": "Unknown classification model."}), 400
            bundle = load_bundle(CLASSIFICATION_MODELS[model_key]["pickle"])
            model = bundle["model"]
            le = bundle["label_encoder"]
            if "scaler" in bundle and bundle["scaler"] is not None:
                X_proc = bundle["scaler"].transform(X_input)
            else:
                X_proc = X_input.values
            pred_idx = int(model.predict(X_proc)[0])
            pred_label = le.inverse_transform([pred_idx])[0]

            proba = None
            if hasattr(model, "predict_proba"):
                proba_arr = model.predict_proba(X_proc)[0]
                proba = {le.inverse_transform([i])[0]: round(float(p), 3) for i, p in enumerate(proba_arr)}

            return jsonify({"prediction": pred_label, "label": "Predicted Performance", "probabilities": proba})

        return jsonify({"error": "Invalid task type."}), 400
    except Exception as exc:
        app.logger.exception("Prediction failed for model=%s task=%s", model_key, task)
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/analyze")
def analyze_page():
    return render_template(
        "analyze.html",
        reg_models=REGRESSION_MODELS,
        cls_models=CLASSIFICATION_MODELS,
    )


@app.route("/api/analyze/<model_key>")
def api_analyze(model_key):
    if model_key not in ANALYSIS_CONFIG:
        return jsonify({"error": "Unknown model."}), 404

    config = ANALYSIS_CONFIG[model_key]
    kind = config["kind"]
    result_bucket = "regression" if kind == "regression" else "classification"
    result_data = RESULTS[result_bucket].get(model_key, {})

    return jsonify({
        "label": config["label"],
        "kind": kind,
        "metrics": result_data.get("metrics", {}),
        "params": clean_params(result_data.get("params", {})),
        "feature_engineering": result_data.get("feature_engineering", ""),
        "top3_features": result_data.get("top3_features", []),
        "plots": [{"file": p, "desc": d} for p, d in config["plots"]],
        "importance_plot": f"{model_key}_top3.png",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
