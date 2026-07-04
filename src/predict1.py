import joblib
import pandas as pd

# Load pipeline
model = joblib.load("models/model_pipeline.pkl")

# Sample data (no scaling needed!)
sample = pd.DataFrame([{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}])

# Predict
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("Prediction:", prediction)
print("Confidence:", probability)
