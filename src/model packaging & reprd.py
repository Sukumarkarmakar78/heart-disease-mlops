# ============================
# IMPORTS
# ============================
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ============================
# LOAD DATA
# ============================
df = pd.read_csv("data/processed/heart_cleaned.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Identify numeric columns
numeric_cols = ["age","trestbps","chol","thalach","oldpeak"]

# ============================
# PREPROCESSING PIPELINE
# ============================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols)
    ],
    remainder="passthrough"
)

# ============================
# FULL PIPELINE (IMPORTANT)
# ============================

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    ))
])

# ============================
# TRAIN TEST SPLIT
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# MLflow EXPERIMENT
# ============================

mlflow.set_experiment("Heart Disease Packaging")

with mlflow.start_run():

    # Train
    model_pipeline.fit(X_train, y_train)

    # Predict
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:,1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    # Log parameters
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", roc)

    # ============================
    # SAVE MODEL (MLflow + Pickle)
    # ============================

    # MLflow format
    mlflow.sklearn.log_model(model_pipeline, "model")

    # Pickle format (local reuse)
    joblib.dump(model_pipeline, "models/model_pipeline.pkl")

    print("✅ Model & pipeline saved successfully!")
