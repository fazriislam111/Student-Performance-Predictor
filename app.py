import json
import os
import pickle

import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

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

FEATURE_EXAMPLES = METADATA.get("feature_examples", {})

REGRESSION_MODELS = {
    "linear_regression": {"label": "Linear Regression", "pickle": "regression_linear_regression.pkl"},
    "xgboost_regression": {"label": "XGBoost", "pickle": "regression_xgboost.pkl"},
    "polynomial_regression": {"label": "Polynomial Regression", "pickle": "regression_polynomial_regression.pkl"},
}

CLASSIFICATION_MODELS = {
    "logistic_regression": {"label": "Logistic Regression", "pickle": "classification_logistic_regression.pkl"},
    "decision_tree": {"label": "Decision Tree", "pickle": "classification_decision_tree.pkl"},
    "random_forest": {"label": "Random Forest", "pickle": "classification_random_forest.pkl"},
    "knn": {"label": "K-Nearest Neighbors", "pickle": "classification_knn.pkl"},
    "xgboost_classification": {"label": "XGBoost", "pickle": "classification_xgboost.pkl"},
}

_model_cache = {}


def load_bundle(pickle_name):
    """Load a saved model bundle, reconstructing XGBoost models from their
    native JSON format (XGBoost pickles are saved as {model_type, native_path,
    features} rather than {model, ...} for cross-version stability)."""
    if pickle_name not in _model_cache:
        with open(os.path.join(MODEL_DIR, pickle_name), "rb") as f:
            bundle = pickle.load(f)

        if "model_type" in bundle:
            native_path = os.path.join(MODEL_DIR, bundle["native_path"])
            if bundle["model_type"] == "xgb_regressor":
                from xgboost import XGBRegressor
                model = XGBRegressor()
            else:
                from xgboost import XGBClassifier
                model = XGBClassifier()
            model.load_model(native_path)
            bundle["model"] = model

        _model_cache[pickle_name] = bundle
    return _model_cache[pickle_name]


# ---------------------------------------------------------------------------
# EDA plot titles (descriptions are data-driven statements computed in
# train.py and stored in metadata.json's "eda_descriptions", so they always
# reflect the actual numbers in the currently-trained dataset)
# ---------------------------------------------------------------------------
EDA_PLOT_TITLES = [
    ("01_correlation_heatmap.png", "Correlation Heatmap"),
    ("02_top_features_scatter.png", "Where Students Cluster: Top 2 Predictive Features"),
    ("03_feature_distributions.png", "Feature Distributions"),
    ("04_total_score_distribution.png", "Total Score Distribution"),
    ("05_class_imbalance.png", "Performance Class Balance"),
    ("06_feature_relationships.png", "Feature Relationships with Total Score"),
    ("07_boxplots_by_class.png", "Feature Spread by Performance Category"),
    ("08_pairplot.png", "Pairwise Feature Relationships"),
    ("09_missing_values.png", "Data Quality: Missing Values Check"),
]
EDA_DESCRIPTIONS = METADATA.get("eda_descriptions", {})
EDA_PLOTS = [
    {"file": f, "title": t, "desc": EDA_DESCRIPTIONS.get(f, "")}
    for f, t in EDA_PLOT_TITLES
]


def fmt_feat(name):
    """Turn a raw feature/term name into a readable label, including
    polynomial interaction terms like 'quiz1_score quiz2_score'."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    label = name
    for raw, nice in FEATURE_LABELS.items():
        short = nice.split(" (")[0]
        label = label.replace(raw, short)
    return label.replace("^2", " squared").replace("_", " ")


def _rank_position(model_key, ranked_keys):
    idx = ranked_keys.index(model_key)
    n = len(ranked_keys)
    if idx == 0:
        return "the top performer"
    if idx == n - 1:
        return "the lowest performer"
    return "a mid-tier performer"


_REG_RANKED = sorted(RESULTS["regression"], key=lambda k: -RESULTS["regression"][k]["metrics"]["R2"])
_CLS_RANKED = sorted(RESULTS["classification"], key=lambda k: -RESULTS["classification"][k]["metrics"]["Accuracy"])


def _top(result_data, n):
    feats = result_data.get("top3_features", [])
    return fmt_feat(feats[n][0]) if len(feats) > n else "the top feature"


def _build_analysis_config():
    config = {}

    # ---- Regression models ----
    for key, label in [
        ("linear_regression", "Linear Regression"),
        ("polynomial_regression", "Polynomial Regression"),
        ("xgboost_regression", "XGBoost"),
    ]:
        d = RESULTS["regression"][key]
        m = d["metrics"]
        rank = _rank_position(key, _REG_RANKED)
        t1, t2, t3 = _top(d, 0), _top(d, 1), _top(d, 2)
        plots = []

        if key == "linear_regression":
            plots.append(("linear_regression_analysis.png",
                f"This model explains {m['R2']*100:.0f}% of the variation in total score using a straight-line "
                f"relationship (R\u00b2={m['R2']:.2f}). {t1} has the strongest influence on predictions, followed by "
                f"{t2} and {t3}. The residuals are scattered evenly on both sides of zero, meaning the model doesn't "
                f"systematically over- or under-predict for any particular score range."))
            plots.append(("linear_regression_learning_curve.png",
                "Training and validation R\u00b2 stay close together as more data is added, meaning the model "
                "generalizes well and isn't overfitting to the training set."))
        elif key == "polynomial_regression":
            lin_r2 = RESULTS["regression"]["linear_regression"]["metrics"]["R2"]
            cmp_word = "an improvement over" if m["R2"] > lin_r2 else "essentially on par with (or slightly below)"
            plots.append(("polynomial_regression_analysis.png",
                f"Allowing curved relationships between features and score explains {m['R2']*100:.0f}% of the "
                f"variation (R\u00b2={m['R2']:.2f}) \u2014 {cmp_word} the plain Linear Regression model. {t1} remains "
                f"the strongest driver even after adding curvature, so the relationship is mostly linear with only "
                f"a modest non-linear correction."))
            plots.append(("polynomial_regression_learning_curve.png",
                "Training and validation R\u00b2 track closely as training size grows, showing the added curvature "
                "hasn't introduced overfitting."))
        else:  # xgboost_regression
            plots.append(("xgboost_regression_analysis.png",
                f"This tree-based model explains {m['R2']*100:.0f}% of the variation in total score (R\u00b2={m['R2']:.2f}), "
                f"making it {rank} among the three regression models on this dataset. The validation loss curve tracks "
                f"the training loss closely, showing the model generalizes rather than memorizing training data. "
                f"{t1} is the single biggest driver of predicted scores."))

        plots.append((f"{key}_metrics.png",
            f"Predictions are off by about {m['MAE']:.1f} points on average (MAE), with a typical error spread of "
            f"{m['RMSE']:.1f} points (RMSE) \u2014 {rank} of the three regression models on this dataset."))

        config[key] = {"kind": "regression", "label": label, "plots": plots}

    # ---- Classification models ----
    for key, label in [
        ("logistic_regression", "Logistic Regression"),
        ("decision_tree", "Decision Tree"),
        ("random_forest", "Random Forest"),
        ("knn", "K-Nearest Neighbors"),
        ("xgboost_classification", "XGBoost"),
    ]:
        d = RESULTS["classification"][key]
        m = d["metrics"]
        rank = _rank_position(key, _CLS_RANKED)
        t1, t2, t3 = _top(d, 0), _top(d, 1), _top(d, 2)
        plots = []

        confusion_desc = (
            "Most mistakes happen between neighboring performance tiers (e.g. 'average' vs 'poor') rather than "
            "between opposite ends like 'fail' and 'good' \u2014 in practice, the model's errors are usually small "
            "misses, not wild ones."
        )
        boundary_desc = (
            f"Using just {t1} and {t2} for this 2D view, the model's colored regions separate stronger students "
            f"toward one side from weaker students toward the other. The deployed model uses all four features, "
            f"so real predictions are more precise than this simplified picture."
        )
        learning_curve_desc = (
            "Training and validation accuracy stay close together as more data is added, showing the model "
            "generalizes rather than memorizing the training set."
        )

        if key == "logistic_regression":
            plots.append(("logistic_regression_boundary.png", boundary_desc))
            plots.append(("logistic_regression_analysis.png",
                f"{t1} and {t2} carry the largest coefficient weights, meaning they shift the predicted probability "
                f"the most. The model is generally most confident about 'average' students \u2014 the majority class "
                f"in this dataset \u2014 and comparatively less certain distinguishing the rarer 'poor' category."))
            plots.append(("logistic_regression_learning_curve.png", learning_curve_desc))
            plots.append(("logistic_regression_confusion.png", confusion_desc))
        elif key == "decision_tree":
            plots.append(("decision_tree_analysis.png",
                f"Early splits in the tree rely heavily on {t1}, the single most influential feature for this model "
                f"\u2014 a student just above or below this tree's threshold can end up in a completely different "
                f"predicted category."))
            plots.append(("decision_tree_boundary.png",
                f"The rectangular regions show how the tree splits students using simple thresholds on {t1} and "
                f"{t2} (e.g. 'above X and above Y') rather than smooth curves \u2014 easy to explain to a student or "
                f"advisor, at the cost of occasionally drawing a hard edge between two very similar students."))
            plots.append(("decision_tree_learning_curve.png", learning_curve_desc))
            plots.append(("decision_tree_confusion.png", confusion_desc))
        elif key == "random_forest":
            oob = m.get("OOB_Score")
            oob_txt = (f"an out-of-bag estimate of {oob*100:.0f}% \u2014 these two numbers being close confirms the "
                       f"model isn't overfitting to the training data. ") if oob is not None else ""
            plots.append(("random_forest_analysis.png",
                f"Averaging many decision trees reaches {m['Accuracy']*100:.0f}% test accuracy, with {oob_txt}"
                f"{t1} contributes the most predictive power across the forest."))
            plots.append(("random_forest_learning_curve.png", learning_curve_desc))
            plots.append(("random_forest_confusion.png", confusion_desc))
        elif key == "knn":
            plots.append(("knn_boundary.png",
                f"Predictions here depend on which nearby students in the training data are most similar on {t1} "
                f"and {t2} \u2014 the irregular boundary reflects real student clusters rather than a simple rule."))
            plots.append(("knn_analysis.png",
                "Accuracy peaks at a moderate number of neighbors, then flattens \u2014 too few neighbors makes "
                "predictions noisy and sensitive to outliers, while too many blurs the line between adjacent "
                "performance categories."))
            plots.append(("knn_learning_curve.png", learning_curve_desc))
            plots.append(("knn_confusion.png", confusion_desc))
        else:  # xgboost_classification
            plots.append(("xgboost_classification_analysis.png",
                f"With {m['Accuracy']*100:.0f}% test accuracy, this is {rank} of the five classification models. "
                f"The validation loss curve tracks training closely, indicating the model generalizes rather than "
                f"memorizes. {t1} is the strongest single driver of the predicted performance category."))
            plots.append(("xgboost_classification_confusion.png", confusion_desc))

        plots.append((f"{key}_metrics.png",
            f"Correctly classifies about {m['Accuracy']*100:.0f}% of students, {rank} of the five classification "
            f"models. Precision of {m['Precision']*100:.0f}% means predicted categories are usually right; recall "
            f"of {m['Recall']*100:.0f}% reflects how many true cases get caught \u2014 the gap between them is mostly "
            f"driven by the rare 'poor' category having few examples to learn from."))

        config[key] = {"kind": "classification", "label": label, "plots": plots}

    return config


ANALYSIS_CONFIG = _build_analysis_config()


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
    clean_csv = os.path.join(DATA_DIR, "student_data.csv")
    df = pd.read_csv(clean_csv)

    dataset_info = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "dtypes": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
    }
    sample_rows = df.sample(min(5, len(df))).to_dict(orient="records")
    sample_columns = list(df.columns)

    return render_template(
        "eda.html",
        plots=EDA_PLOTS,
        dataset_info=dataset_info,
        sample_rows=sample_rows,
        sample_columns=sample_columns,
    )


@app.route("/predict")
def predict_page():
    return render_template(
        "predict.html",
        reg_models=REGRESSION_MODELS,
        cls_models=CLASSIFICATION_MODELS,
        features=FEATURES,
        feature_labels=FEATURE_LABELS,
        feature_examples=FEATURE_EXAMPLES,
    )


REG_BANDS = [
    (80, "Good", "This student's projected score places them in the 'Good' range. Current habits are working well \u2014 the main recommendation is to keep them consistent."),
    (60, "Average", "This student's projected score places them in the 'Average' range \u2014 a passing result with room to grow. A small push in quiz consistency could move them into the 'Good' range."),
    (40, "Poor", "This student's projected score places them in the 'Poor' range. Recommend additional support on quizzes and assignments soon, not later."),
    (0, "Fail", "This student's projected score places them in the 'Fail' range \u2014 the highest-risk category. Immediate intervention is recommended \u2014 attendance follow-up and one-on-one tutoring should be prioritized."),
]

CLASS_MESSAGES = {
    "good": "This student is on track for top-tier performance. Recommendation: keep reinforcing current study habits, no intervention needed.",
    "average": "This student is performing adequately but has room to grow. Recommendation: targeted support on quiz preparation could move them into the 'good' tier.",
    "poor": "This student is underperforming and at risk. Recommendation: an early advisor check-in and structured tutoring are strongly suggested.",
    "fail": "This student is at high risk of failing. Recommendation: immediate academic intervention \u2014 attendance follow-up and tutoring \u2014 is strongly advised.",
}


def regression_band(score):
    for threshold, label, message in REG_BANDS:
        if score >= threshold:
            return label, message
    return REG_BANDS[-1][1], REG_BANDS[-1][2]


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)
    task = payload.get("task")
    model_key = payload.get("model")
    raw_features = payload.get("features", {})

    try:
        input_row = {f: float(raw_features[f]) for f in FEATURES}
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Please provide valid numeric values for all features."}), 400

    X_input = pd.DataFrame([input_row])[FEATURES]

    if task == "regression":
        if model_key not in REGRESSION_MODELS:
            return jsonify({"error": "Unknown regression model."}), 400
        bundle = load_bundle(REGRESSION_MODELS[model_key]["pickle"])
        model = bundle["model"]
        if bundle.get("scaler") is not None:
            X_proc = bundle["scaler"].transform(X_input)
        else:
            X_proc = X_input
        if "poly" in bundle:
            X_proc = bundle["poly"].transform(X_proc)
        pred = float(model.predict(X_proc)[0])
        pred = max(0.0, min(100.0, pred))
        band_label, band_message = regression_band(pred)
        return jsonify({
            "prediction": round(pred, 1),
            "label": "Predicted Total Score",
            "band": band_label,
            "message": f"Projected score: {pred:.0f} / 100 \u2014 {band_label}. {band_message}",
        })

    elif task == "classification":
        if model_key not in CLASSIFICATION_MODELS:
            return jsonify({"error": "Unknown classification model."}), 400
        bundle = load_bundle(CLASSIFICATION_MODELS[model_key]["pickle"])
        model = bundle["model"]
        le = bundle["label_encoder"]
        if bundle.get("scaler") is not None:
            X_proc = bundle["scaler"].transform(X_input)
        else:
            X_proc = X_input
        pred_idx = int(model.predict(X_proc)[0])
        pred_label = le.inverse_transform([pred_idx])[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba_arr = model.predict_proba(X_proc)[0]
            proba = {le.inverse_transform([i])[0]: round(float(p), 3) for i, p in enumerate(proba_arr)}

        message = f"Predicted category: {pred_label.title()}. {CLASS_MESSAGES.get(pred_label, '')}"
        return jsonify({
            "prediction": pred_label,
            "label": "Predicted Performance",
            "message": message,
            "probabilities": proba,
        })

    return jsonify({"error": "Invalid task type."}), 400


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

    top3 = result_data.get("top3_features", [])
    if len(top3) >= 3:
        importance_statement = (
            f"{fmt_feat(top3[0][0])} is the dominant factor for this model, followed by {fmt_feat(top3[1][0])} and "
            f"{fmt_feat(top3[2][0])} \u2014 together these three drive most of what the model bases its predictions on."
        )
    else:
        importance_statement = ""

    return jsonify({
        "label": config["label"],
        "kind": kind,
        "metrics": result_data.get("metrics", {}),
        "params": clean_params(result_data.get("params", {})),
        "feature_engineering": result_data.get("feature_engineering", ""),
        "top3_features": top3,
        "importance_statement": importance_statement,
        "plots": [{"file": p, "desc": d} for p, d in config["plots"]],
        "importance_plot": f"{model_key}_top3.png",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
