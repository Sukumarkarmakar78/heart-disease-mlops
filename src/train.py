# ============================
# IMPORTS

# ============================
import os

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================
# LOAD CLEAN DATA
# ============================
df = pd.read_csv("data/processed/heart_cleaned.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# START MLflow EXPERIMENT
# ============================
mlflow.set_experiment("Heart Disease Prediction")

with mlflow.start_run():

    # ============================
    # MODEL 1: Logistic Regression
    # ============================
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    y_pred_log = log_model.predict(X_test)
    y_prob_log = log_model.predict_proba(X_test)[:, 1]

    # Metrics
    acc_log = accuracy_score(y_test, y_pred_log)
    roc_log = roc_auc_score(y_test, y_prob_log)

    # Log parameters + metrics
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("log_accuracy", acc_log)
    mlflow.log_metric("log_roc_auc", roc_log)

    # ============================
    # MODEL 2: Random Forest
    # ============================
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["age", "trestbps", "chol", "thalach", "oldpeak"])
        ],
        remainder="passthrough"
    )
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10],
    }

    grid = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring="roc_auc")
    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_

    y_pred_rf = best_pipeline.predict(X_test)
    y_prob_rf = best_pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf)
    recall_rf = recall_score(y_test, y_pred_rf)
    roc_rf = roc_auc_score(y_test, y_prob_rf)

    # Log parameters
    mlflow.log_param("rf_n_estimators", grid.best_params_["classifier__n_estimators"])
    mlflow.log_param("rf_max_depth", grid.best_params_["classifier__max_depth"])

    # Log metrics
    mlflow.log_metric("rf_accuracy", acc_rf)
    mlflow.log_metric("rf_precision", precision_rf)
    mlflow.log_metric("rf_recall", recall_rf)
    mlflow.log_metric("rf_roc_auc", roc_rf)

    # ============================
    # SAVE MODEL
    # ============================
    mlflow.sklearn.log_model(best_pipeline, "model")

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_pipeline, os.path.join(model_dir, "model_pipeline.pkl"))

    # ============================
    # LOG PLOT (ROC CURVE)
    # ============================
    fpr, tpr, _ = roc_curve(y_test, y_prob_rf)

    plt.figure()
    plt.plot(fpr, tpr, label=f"RF ROC (AUC={roc_rf:.2f})")
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")

    # ============================
    # LOG FEATURE IMPORTANCE
    # ============================
    feature_importance = best_rf.feature_importances_

    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importance, y=X.columns)
    plt.title("Feature Importance")
    plt.savefig("feature_importance.png")

    mlflow.log_artifact("feature_importance.png")

    print("✅ MLflow tracking complete!")
