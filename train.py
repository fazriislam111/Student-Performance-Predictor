"""
Student Performance Prediction - Model Training Pipeline
Trains 3 regression models + 5 classification models, generates EDA,
per-model analysis plots, feature-importance plots, and saves models.

Retrained on the updated dataset (2000 rows) provided September 2026.
Column names differ slightly from the original dataset; they are renamed
below to the same internal names used throughout app.py, so the web app
requires no code changes to work with this retrained set of models.
"""
import json
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

RANDOM_STATE = 42
RAW_DATA_PATH = "data/student_data_raw.csv"
CLEAN_DATA_PATH = "data/student_data.csv"
EDA_DIR = "plots/eda"
ANALYSIS_DIR = "plots/analysis"
IMPORTANCE_DIR = "plots/importance"
MODEL_DIR = "models"

FEATURES = ["Attendance_Rate", "quiz1_score", "quiz2_score", "assignment_score"]
REG_TARGET = "total_score"
CLS_TARGET = "performance"
CLASS_ORDER = ["fail", "poor", "average", "good"]

# ---------------------------------------------------------------------------
# Load & clean data
# ---------------------------------------------------------------------------
df_raw = pd.read_csv(RAW_DATA_PATH)

# This dataset's raw column names differ from the original project's raw
# columns, but map onto the exact same underlying features. Renaming here
# keeps every downstream name (FEATURES, REG_TARGET, ...) identical to the
# first training run, so app.py needs zero changes to serve these models.
df = df_raw.rename(columns={
    "student_ID": "Student_ID",
    "attendance (%)": "Attendance_Rate",
    "quiz_1 score (out of 15)": "quiz1_score",
    "quiz_2 score (out of 15)": "quiz2_score",
    "assignment score (out of 10)": "assignment_score",
    "final score (out of 100)": REG_TARGET,
})

missing_before = df.isna().sum()
df = df.dropna().reset_index(drop=True)
df.to_csv(CLEAN_DATA_PATH, index=False)

metadata = {
    "features": FEATURES,
    "regression_target": REG_TARGET,
    "classification_target": CLS_TARGET,
    "class_order": CLASS_ORDER,
    "feature_ranges": {f: [float(df[f].min()), float(df[f].max())] for f in FEATURES},
    "n_rows": int(len(df)),
}

X = df[FEATURES].copy()
y_reg = df[REG_TARGET].copy()

le = LabelEncoder()
le.fit(CLASS_ORDER)
y_cls = le.transform(df[CLS_TARGET])

X_train, X_test, yreg_train, yreg_test, ycls_train, ycls_test = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cls
)

results = {"regression": {}, "classification": {}}

# ===========================================================================
# PART 1: EXPLORATORY DATA ANALYSIS
# ===========================================================================
print("Generating EDA plots...")

# 1. Feature distributions
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, feat in zip(axes.flat, FEATURES):
    sns.histplot(df[feat], kde=True, ax=ax, color="#4C72B0")
    ax.set_title(f"Distribution of {feat}")
fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{EDA_DIR}/01_feature_distributions.png")
plt.close(fig)

# 2. Target distribution (total score)
fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(df[REG_TARGET], kde=True, ax=ax, color="#55A868")
ax.set_title("Distribution of Total Score")
fig.tight_layout()
fig.savefig(f"{EDA_DIR}/02_total_score_distribution.png")
plt.close(fig)

# 3. Class imbalance
fig, ax = plt.subplots(figsize=(7, 5))
counts = df[CLS_TARGET].value_counts().reindex(CLASS_ORDER)
sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="viridis", ax=ax, legend=False)
ax.set_title("Class Distribution: Performance Categories")
ax.set_ylabel("Count")
for i, v in enumerate(counts.values):
    ax.text(i, v + 3, str(v), ha="center")
fig.tight_layout()
fig.savefig(f"{EDA_DIR}/03_class_imbalance.png")
plt.close(fig)

# 4. Correlation heatmap
fig, ax = plt.subplots(figsize=(7, 6))
corr = df[FEATURES + [REG_TARGET]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Heatmap")
fig.tight_layout()
fig.savefig(f"{EDA_DIR}/04_correlation_heatmap.png")
plt.close(fig)

# 5. Feature relationships with target (scatter)
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, feat in zip(axes.flat, FEATURES):
    sns.scatterplot(x=df[feat], y=df[REG_TARGET], hue=df[CLS_TARGET],
                     hue_order=CLASS_ORDER, palette="viridis", ax=ax, s=18, legend=False, alpha=0.6)
    ax.set_title(f"{feat} vs Total Score")
fig.suptitle("Feature Relationships with Total Score", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{EDA_DIR}/05_feature_relationships.png")
plt.close(fig)

# 6. Boxplots of features by performance class
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, feat in zip(axes.flat, FEATURES):
    sns.boxplot(x=df[CLS_TARGET], y=df[feat], order=CLASS_ORDER, hue=df[CLS_TARGET],
                hue_order=CLASS_ORDER, palette="Set2", ax=ax, legend=False)
    ax.set_title(f"{feat} by Performance Category")
fig.suptitle("Feature Spread Across Performance Categories", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{EDA_DIR}/06_boxplots_by_class.png")
plt.close(fig)

# 7. Pairplot
pp = sns.pairplot(df[FEATURES + [CLS_TARGET]], hue=CLS_TARGET, hue_order=CLASS_ORDER,
                   palette="viridis", diag_kind="kde", plot_kws={"s": 10, "alpha": 0.5})
pp.fig.suptitle("Pairwise Feature Relationships", y=1.02, fontsize=14, fontweight="bold")
pp.savefig(f"{EDA_DIR}/07_pairplot.png")
plt.close(pp.fig)

# 8. Missing values check
fig, ax = plt.subplots(figsize=(8, 4.5))
all_cols = list(df_raw.columns)
missing_counts = df_raw.isna().sum()
bars = ax.bar(all_cols, missing_counts.values, color="#C44E52")
ax.set_title(f"Missing Values per Column (dataset: {len(df_raw)} rows)")
ax.set_ylabel("Missing count")
ax.set_ylim(0, max(1, missing_counts.max() * 1.3))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
for bar, val in zip(bars, missing_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02 * ax.get_ylim()[1], str(val), ha="center")
fig.tight_layout()
fig.savefig(f"{EDA_DIR}/08_missing_values.png")
plt.close(fig)

print("EDA plots done.")

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def save_sklearn_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_xgb_bundle(xgb_model, extra, native_filename, path):
    """Save XGBoost models via their native JSON format (recommended by
    XGBoost for cross-version stability) instead of raw pickling the
    Booster. The bundle records model_type + native_path so app.py can
    reconstruct the estimator; native model saved alongside as its own file."""
    xgb_model.save_model(f"{MODEL_DIR}/{native_filename}")
    bundle = {
        "model_type": "xgb_regressor" if isinstance(xgb_model, XGBRegressor) else "xgb_classifier",
        "native_path": native_filename,
        **extra,
    }
    save_sklearn_pickle(bundle, path)


from sklearn.model_selection import learning_curve as _sk_learning_curve


def learning_curve_plot(estimator, X, y, title, path, scoring):
    """Plot training vs. validation score across increasing training-set
    sizes -- the standard 'learning curve' diagnostic for models (like
    Linear/Logistic Regression, Decision Trees, Random Forest, KNN) that
    don't train iteratively and so have no per-epoch loss curve."""
    train_sizes, train_scores, val_scores = _sk_learning_curve(
        estimator, X, y, cv=5, scoring=scoring,
        train_sizes=np.linspace(0.2, 1.0, 6), random_state=RANDOM_STATE
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(train_sizes, train_mean, marker="o", label="Training score")
    ax.plot(train_sizes, val_mean, marker="o", label="Validation score")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(scoring.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def top3_importance_plot(names, importances, title, path):
    order = np.argsort(importances)[::-1][:3]
    top_names = [names[i] for i in order]
    top_vals = [importances[i] for i in order]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.barplot(x=top_vals, y=top_names, hue=top_names, palette="mako", ax=ax, legend=False)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return list(zip(top_names, [float(v) for v in top_vals]))


def metrics_bar_plot(metrics_dict, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=list(metrics_dict.keys()), y=list(metrics_dict.values()),
                hue=list(metrics_dict.keys()), palette="crest", ax=ax, legend=False)
    ax.set_title(title)
    for i, v in enumerate(metrics_dict.values()):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

# ===========================================================================
# PART 2: REGRESSION PIPELINE  (target: total_score)
# ===========================================================================
print("Training regression models...")

reg_scaler = StandardScaler()
Xr_train_s = reg_scaler.fit_transform(X_train)
Xr_test_s = reg_scaler.transform(X_test)

def reg_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

# --- 2.1 Linear Regression ---------------------------------------------------
lin_reg = LinearRegression()
lin_reg.fit(Xr_train_s, yreg_train)
learning_curve_plot(LinearRegression(), Xr_train_s, yreg_train,
                     "Linear Regression - Learning Curve",
                     f"{ANALYSIS_DIR}/linear_regression_learning_curve.png", scoring="r2")
lin_pred = lin_reg.predict(Xr_test_s)
lin_metrics = reg_metrics(yreg_test, lin_pred)
results["regression"]["linear_regression"] = {
    "metrics": lin_metrics,
    "params": lin_reg.get_params(),
    "feature_engineering": "StandardScaler applied to all 4 numeric features.",
}

residuals = yreg_test - lin_pred
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].scatter(lin_pred, residuals, alpha=0.5, color="#4C72B0")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted Total Score"); axes[0].set_ylabel("Residual")
axes[0].set_title("Residual Plot")

axes[1].scatter(yreg_test, lin_pred, alpha=0.5, color="#55A868")
lims = [min(yreg_test.min(), lin_pred.min()), max(yreg_test.max(), lin_pred.max())]
axes[1].plot(lims, lims, "r--")
axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
axes[1].set_title("Predicted vs Actual")

coef_series = pd.Series(lin_reg.coef_, index=FEATURES).sort_values()
axes[2].barh(coef_series.index, coef_series.values, color="#C44E52")
axes[2].set_title("Coefficient Importance")
axes[2].set_xlabel("Coefficient (standardized)")

fig.suptitle(f"Linear Regression Analysis  (R²={lin_metrics['R2']:.3f}, RMSE={lin_metrics['RMSE']:.2f})",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{ANALYSIS_DIR}/linear_regression_analysis.png")
plt.close(fig)

metrics_bar_plot(lin_metrics, "Linear Regression - Test Metrics", f"{ANALYSIS_DIR}/linear_regression_metrics.png")

lr_top3 = top3_importance_plot(FEATURES, np.abs(lin_reg.coef_),
                                "Linear Regression - Top 3 Features",
                                f"{IMPORTANCE_DIR}/linear_regression_top3.png")
results["regression"]["linear_regression"]["top3_features"] = lr_top3

save_sklearn_pickle({"model": lin_reg, "scaler": reg_scaler, "features": FEATURES},
                     f"{MODEL_DIR}/regression_linear_regression.pkl")

# --- 2.2 Polynomial Regression ----------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
Xr_train_poly = poly.fit_transform(Xr_train_s)
Xr_test_poly = poly.transform(Xr_test_s)

poly_reg = LinearRegression()
poly_reg.fit(Xr_train_poly, yreg_train)
learning_curve_plot(LinearRegression(), Xr_train_poly, yreg_train,
                     "Polynomial Regression - Learning Curve",
                     f"{ANALYSIS_DIR}/polynomial_regression_learning_curve.png", scoring="r2")
poly_pred = poly_reg.predict(Xr_test_poly)
poly_metrics = reg_metrics(yreg_test, poly_pred)
results["regression"]["polynomial_regression"] = {
    "metrics": poly_metrics,
    "params": {"degree": 2, **poly_reg.get_params()},
    "feature_engineering": "StandardScaler + PolynomialFeatures(degree=2) on all 4 features.",
}

poly_residuals = yreg_test - poly_pred
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(X_test["Attendance_Rate"], yreg_test, alpha=0.4, label="Actual", color="#4C72B0", s=15)
axes[0].scatter(X_test["Attendance_Rate"], poly_pred, alpha=0.4, label="Predicted", color="#DD8452", s=15)
axes[0].set_xlabel("Attendance Rate"); axes[0].set_ylabel("Total Score")
axes[0].set_title("Polynomial Fit Visualization"); axes[0].legend()

axes[1].scatter(poly_pred, poly_residuals, alpha=0.5, color="#C44E52")
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
axes[1].set_title("Residual Analysis")

fig.suptitle(f"Polynomial Regression Analysis (deg=2)  (R²={poly_metrics['R2']:.3f}, RMSE={poly_metrics['RMSE']:.2f})",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{ANALYSIS_DIR}/polynomial_regression_analysis.png")
plt.close(fig)

metrics_bar_plot(poly_metrics, "Polynomial Regression - Test Metrics", f"{ANALYSIS_DIR}/polynomial_regression_metrics.png")

poly_feat_names = poly.get_feature_names_out(FEATURES)
poly_top3 = top3_importance_plot(list(poly_feat_names), np.abs(poly_reg.coef_),
                                  "Polynomial Regression - Top 3 Terms",
                                  f"{IMPORTANCE_DIR}/polynomial_regression_top3.png")
results["regression"]["polynomial_regression"]["top3_features"] = poly_top3

save_sklearn_pickle({"model": poly_reg, "scaler": reg_scaler, "poly": poly, "features": FEATURES},
                     f"{MODEL_DIR}/regression_polynomial_regression.pkl")

# --- 2.3 XGBoost Regressor ---------------------------------------------------
xgb_reg = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
                        eval_metric="rmse")
eval_set = [(X_train, yreg_train), (X_test, yreg_test)]
xgb_reg.fit(X_train, yreg_train, eval_set=eval_set, verbose=False)
xgb_pred = xgb_reg.predict(X_test)
xgb_metrics = reg_metrics(yreg_test, xgb_pred)
results["regression"]["xgboost_regression"] = {
    "metrics": xgb_metrics,
    "params": xgb_reg.get_params(),
    "feature_engineering": "None required (tree-based); raw numeric features used directly.",
}

evals_result = xgb_reg.evals_result()
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].plot(evals_result["validation_0"]["rmse"], label="Train")
axes[0].plot(evals_result["validation_1"]["rmse"], label="Validation")
axes[0].set_xlabel("Boosting Round"); axes[0].set_ylabel("RMSE")
axes[0].set_title("Training/Validation Loss Curve"); axes[0].legend()

importances = xgb_reg.feature_importances_
sorted_idx = np.argsort(importances)
axes[1].barh(np.array(FEATURES)[sorted_idx], importances[sorted_idx], color="#4C72B0")
axes[1].set_title("Feature Importance (Gain-based)")

axes[2].scatter(yreg_test, xgb_pred, alpha=0.5, color="#55A868")
lims = [min(yreg_test.min(), xgb_pred.min()), max(yreg_test.max(), xgb_pred.max())]
axes[2].plot(lims, lims, "r--")
axes[2].set_xlabel("Actual"); axes[2].set_ylabel("Predicted")
axes[2].set_title("Predicted vs Actual")

fig.suptitle(f"XGBoost Regression Analysis  (R²={xgb_metrics['R2']:.3f}, RMSE={xgb_metrics['RMSE']:.2f})",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{ANALYSIS_DIR}/xgboost_regression_analysis.png")
plt.close(fig)

metrics_bar_plot(xgb_metrics, "XGBoost Regression - Test Metrics", f"{ANALYSIS_DIR}/xgboost_regression_metrics.png")

xgbr_top3 = top3_importance_plot(FEATURES, importances,
                                  "XGBoost Regression - Top 3 Features",
                                  f"{IMPORTANCE_DIR}/xgboost_regression_top3.png")
results["regression"]["xgboost_regression"]["top3_features"] = xgbr_top3

save_xgb_bundle(xgb_reg, {"features": FEATURES}, "regression_xgboost_native.json",
                 f"{MODEL_DIR}/regression_xgboost.pkl")

print("Regression pipeline complete.")
print(json.dumps({k: v["metrics"] for k, v in results["regression"].items()}, indent=2))

# ===========================================================================
# PART 3: CLASSIFICATION PIPELINE  (target: performance)
# ===========================================================================
print("Training classification models...")

cls_scaler = StandardScaler()
Xc_train_s = cls_scaler.fit_transform(X_train)
Xc_test_s = cls_scaler.transform(X_test)

def cls_metrics(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

def confusion_plot(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_ORDER,
                yticklabels=CLASS_ORDER, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

boundary_feats = ["Attendance_Rate", "quiz1_score"]
Xb_train_2d = X_train[boundary_feats].values
b_scaler = StandardScaler().fit(Xb_train_2d)
Xb_train_2d_s = b_scaler.transform(Xb_train_2d)

def plot_decision_boundary(model_2d, X2d, y2d, title, path, feat_names):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    x_min, x_max = X2d[:, 0].min() - 1, X2d[:, 0].max() + 1
    y_min, y_max = X2d[:, 1].min() - 1, X2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    scatter = ax.scatter(X2d[:, 0], X2d[:, 1], c=y2d, cmap="viridis", edgecolor="k", s=14, linewidths=0.3)
    ax.set_xlabel(feat_names[0] + " (scaled)"); ax.set_ylabel(feat_names[1] + " (scaled)")
    ax.set_title(title)
    handles, _ = scatter.legend_elements()
    ax.legend(handles, CLASS_ORDER, title="Performance")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

# --- 3.1 Logistic Regression --------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(Xc_train_s, ycls_train)
learning_curve_plot(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), Xc_train_s, ycls_train,
                     "Logistic Regression - Learning Curve",
                     f"{ANALYSIS_DIR}/logistic_regression_learning_curve.png", scoring="accuracy")
log_pred = log_reg.predict(Xc_test_s)
log_proba = log_reg.predict_proba(Xc_test_s)
log_metrics = cls_metrics(ycls_test, log_pred)
results["classification"]["logistic_regression"] = {
    "metrics": log_metrics,
    "params": log_reg.get_params(),
    "feature_engineering": "StandardScaler applied to all 4 numeric features.",
}

log_reg_2d = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(Xb_train_2d_s, ycls_train)
plot_decision_boundary(log_reg_2d, Xb_train_2d_s, ycls_train,
                        "Logistic Regression Decision Boundary\n(Attendance_Rate vs quiz1_score)",
                        f"{ANALYSIS_DIR}/logistic_regression_boundary.png", boundary_feats)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
coef_df = pd.DataFrame(log_reg.coef_, columns=FEATURES, index=CLASS_ORDER)
sns.heatmap(coef_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0])
axes[0].set_title("Coefficients per Class")

for i, cls_name in enumerate(CLASS_ORDER):
    sns.kdeplot(log_proba[:, i], ax=axes[1], label=cls_name)
axes[1].set_title("Predicted Probability Distributions")
axes[1].set_xlabel("Predicted Probability"); axes[1].legend()

fig.suptitle(f"Logistic Regression Analysis (Acc={log_metrics['Accuracy']:.3f})", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{ANALYSIS_DIR}/logistic_regression_analysis.png")
plt.close(fig)

confusion_plot(ycls_test, log_pred, "Logistic Regression - Confusion Matrix",
               f"{ANALYSIS_DIR}/logistic_regression_confusion.png")
metrics_bar_plot(log_metrics, "Logistic Regression - Test Metrics", f"{ANALYSIS_DIR}/logistic_regression_metrics.png")

log_importance = np.abs(log_reg.coef_).mean(axis=0)
log_top3 = top3_importance_plot(FEATURES, log_importance,
                                 "Logistic Regression - Top 3 Features",
                                 f"{IMPORTANCE_DIR}/logistic_regression_top3.png")
results["classification"]["logistic_regression"]["top3_features"] = log_top3

save_sklearn_pickle({"model": log_reg, "scaler": cls_scaler, "label_encoder": le, "features": FEATURES},
                     f"{MODEL_DIR}/classification_logistic_regression.pkl")

# --- 3.2 Decision Tree --------------------------------------------------------
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=RANDOM_STATE)
dt.fit(X_train, ycls_train)
learning_curve_plot(DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=RANDOM_STATE),
                     X_train, ycls_train, "Decision Tree - Learning Curve",
                     f"{ANALYSIS_DIR}/decision_tree_learning_curve.png", scoring="accuracy")
dt_pred = dt.predict(X_test)
dt_metrics = cls_metrics(ycls_test, dt_pred)
results["classification"]["decision_tree"] = {
    "metrics": dt_metrics,
    "params": dt.get_params(),
    "feature_engineering": "None required (tree-based); raw numeric features used directly.",
}

fig, ax = plt.subplots(figsize=(16, 8))
plot_tree(dt, feature_names=FEATURES, class_names=CLASS_ORDER, filled=True, fontsize=8, ax=ax, max_depth=3)
ax.set_title(f"Decision Tree Structure (top 3 levels shown, Acc={dt_metrics['Accuracy']:.3f})")
fig.tight_layout()
fig.savefig(f"{ANALYSIS_DIR}/decision_tree_analysis.png")
plt.close(fig)

dt_2d = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=RANDOM_STATE).fit(Xb_train_2d, ycls_train)
plot_decision_boundary(dt_2d, Xb_train_2d, ycls_train,
                        "Decision Tree Decision Boundary\n(Attendance_Rate vs quiz1_score)",
                        f"{ANALYSIS_DIR}/decision_tree_boundary.png", boundary_feats)

confusion_plot(ycls_test, dt_pred, "Decision Tree - Confusion Matrix",
               f"{ANALYSIS_DIR}/decision_tree_confusion.png")
metrics_bar_plot(dt_metrics, "Decision Tree - Test Metrics", f"{ANALYSIS_DIR}/decision_tree_metrics.png")

dt_top3 = top3_importance_plot(FEATURES, dt.feature_importances_,
                                "Decision Tree - Top 3 Features",
                                f"{IMPORTANCE_DIR}/decision_tree_top3.png")
results["classification"]["decision_tree"]["top3_features"] = dt_top3

save_sklearn_pickle({"model": dt, "label_encoder": le, "features": FEATURES},
                     f"{MODEL_DIR}/classification_decision_tree.pkl")

# --- 3.3 Random Forest --------------------------------------------------------
rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=4,
                             oob_score=True, random_state=RANDOM_STATE)
rf.fit(X_train, ycls_train)
learning_curve_plot(RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=4, random_state=RANDOM_STATE),
                     X_train, ycls_train, "Random Forest - Learning Curve",
                     f"{ANALYSIS_DIR}/random_forest_learning_curve.png", scoring="accuracy")
rf_pred = rf.predict(X_test)
rf_metrics = cls_metrics(ycls_test, rf_pred)
rf_metrics["OOB_Score"] = float(rf.oob_score_)
results["classification"]["random_forest"] = {
    "metrics": rf_metrics,
    "params": rf.get_params(),
    "feature_engineering": "None required (tree-based); raw numeric features used directly.",
}

oob_errors = []
tree_counts = list(range(20, 320, 20))
rf_oob = RandomForestClassifier(warm_start=True, oob_score=True, max_depth=10,
                                 min_samples_leaf=4, random_state=RANDOM_STATE, n_estimators=1)
for n in tree_counts:
    rf_oob.set_params(n_estimators=n)
    rf_oob.fit(X_train, ycls_train)
    oob_errors.append(1 - rf_oob.oob_score_)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(tree_counts, oob_errors, marker="o", color="#4C72B0")
axes[0].set_xlabel("Number of Trees"); axes[0].set_ylabel("OOB Error Rate")
axes[0].set_title("Out-of-Bag Error Analysis")

importances = rf.feature_importances_
sorted_idx = np.argsort(importances)
axes[1].barh(np.array(FEATURES)[sorted_idx], importances[sorted_idx], color="#55A868")
axes[1].set_title("Feature Importance")

fig.suptitle(f"Random Forest Analysis (Acc={rf_metrics['Accuracy']:.3f}, OOB={rf.oob_score_:.3f})",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{ANALYSIS_DIR}/random_forest_analysis.png")
plt.close(fig)

confusion_plot(ycls_test, rf_pred, "Random Forest - Confusion Matrix",
               f"{ANALYSIS_DIR}/random_forest_confusion.png")
metrics_bar_plot({k: v for k, v in rf_metrics.items() if k != "OOB_Score"},
                  "Random Forest - Test Metrics", f"{ANALYSIS_DIR}/random_forest_metrics.png")

rf_top3 = top3_importance_plot(FEATURES, importances,
                                "Random Forest - Top 3 Features",
                                f"{IMPORTANCE_DIR}/random_forest_top3.png")
results["classification"]["random_forest"]["top3_features"] = rf_top3

save_sklearn_pickle({"model": rf, "label_encoder": le, "features": FEATURES},
                     f"{MODEL_DIR}/classification_random_forest.pkl")

# --- 3.4 KNN -------------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(Xc_train_s, ycls_train)
learning_curve_plot(KNeighborsClassifier(n_neighbors=15), Xc_train_s, ycls_train,
                     "KNN - Learning Curve",
                     f"{ANALYSIS_DIR}/knn_learning_curve.png", scoring="accuracy")
knn_pred = knn.predict(Xc_test_s)
knn_metrics = cls_metrics(ycls_test, knn_pred)
results["classification"]["knn"] = {
    "metrics": knn_metrics,
    "params": knn.get_params(),
    "feature_engineering": "StandardScaler applied to all 4 numeric features (distance-based model).",
}

knn_2d = KNeighborsClassifier(n_neighbors=15).fit(Xb_train_2d_s, ycls_train)
plot_decision_boundary(knn_2d, Xb_train_2d_s, ycls_train,
                        "KNN Decision Boundary (k=15)\n(Attendance_Rate vs quiz1_score)",
                        f"{ANALYSIS_DIR}/knn_boundary.png", boundary_feats)

k_values = list(range(1, 35, 2))
k_accuracies = []
for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k).fit(Xc_train_s, ycls_train)
    k_accuracies.append(accuracy_score(ycls_test, knn_k.predict(Xc_test_s)))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(k_values, k_accuracies, marker="o", color="#C44E52")
ax.axvline(15, color="gray", linestyle="--", label="chosen k=15")
ax.set_xlabel("k (Number of Neighbors)"); ax.set_ylabel("Test Accuracy")
ax.set_title("Neighbor Influence: Accuracy vs k")
ax.legend()
fig.tight_layout()
fig.savefig(f"{ANALYSIS_DIR}/knn_analysis.png")
plt.close(fig)

confusion_plot(ycls_test, knn_pred, "KNN - Confusion Matrix", f"{ANALYSIS_DIR}/knn_confusion.png")
metrics_bar_plot(knn_metrics, "KNN - Test Metrics", f"{ANALYSIS_DIR}/knn_metrics.png")

perm = permutation_importance(knn, Xc_test_s, ycls_test, n_repeats=20, random_state=RANDOM_STATE)
knn_top3 = top3_importance_plot(FEATURES, perm.importances_mean,
                                 "KNN - Top 3 Features (Permutation Importance)",
                                 f"{IMPORTANCE_DIR}/knn_top3.png")
results["classification"]["knn"]["top3_features"] = knn_top3

save_sklearn_pickle({"model": knn, "scaler": cls_scaler, "label_encoder": le, "features": FEATURES},
                     f"{MODEL_DIR}/classification_knn.pkl")

# --- 3.5 XGBoost Classifier ----------------------------------------------------
xgb_cls = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
                         eval_metric="mlogloss", objective="multi:softprob", num_class=4)
eval_set_c = [(X_train, ycls_train), (X_test, ycls_test)]
xgb_cls.fit(X_train, ycls_train, eval_set=eval_set_c, verbose=False)
xgb_cls_pred = xgb_cls.predict(X_test)
xgb_cls_metrics = cls_metrics(ycls_test, xgb_cls_pred)
results["classification"]["xgboost_classification"] = {
    "metrics": xgb_cls_metrics,
    "params": xgb_cls.get_params(),
    "feature_engineering": "None required (tree-based); raw numeric features used directly.",
}

evals_result_c = xgb_cls.evals_result()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(evals_result_c["validation_0"]["mlogloss"], label="Train")
axes[0].plot(evals_result_c["validation_1"]["mlogloss"], label="Validation")
axes[0].set_xlabel("Boosting Round"); axes[0].set_ylabel("Log Loss")
axes[0].set_title("Training/Validation Loss Curve"); axes[0].legend()

importances_c = xgb_cls.feature_importances_
sorted_idx = np.argsort(importances_c)
axes[1].barh(np.array(FEATURES)[sorted_idx], importances_c[sorted_idx], color="#8172B2")
axes[1].set_title("Feature Importance (Gain-based)")

fig.suptitle(f"XGBoost Classification Analysis (Acc={xgb_cls_metrics['Accuracy']:.3f})",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{ANALYSIS_DIR}/xgboost_classification_analysis.png")
plt.close(fig)

confusion_plot(ycls_test, xgb_cls_pred, "XGBoost Classification - Confusion Matrix",
               f"{ANALYSIS_DIR}/xgboost_classification_confusion.png")
metrics_bar_plot(xgb_cls_metrics, "XGBoost Classification - Test Metrics",
                  f"{ANALYSIS_DIR}/xgboost_classification_metrics.png")

xgbc_top3 = top3_importance_plot(FEATURES, importances_c,
                                  "XGBoost Classification - Top 3 Features",
                                  f"{IMPORTANCE_DIR}/xgboost_classification_top3.png")
results["classification"]["xgboost_classification"]["top3_features"] = xgbc_top3

save_xgb_bundle(xgb_cls, {"label_encoder": le, "features": FEATURES}, "classification_xgboost_native.json",
                 f"{MODEL_DIR}/classification_xgboost.pkl")

print("Classification pipeline complete.")
print(json.dumps({k: v["metrics"] for k, v in results["classification"].items()}, indent=2))

# ===========================================================================
# SAVE METADATA / RESULTS SUMMARY
# ===========================================================================
with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

with open("models/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("ALL DONE. Models saved to models/, plots saved to plots/.")
