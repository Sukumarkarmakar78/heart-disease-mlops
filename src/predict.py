import numpy as np
import pickle
import pandas as pd

# ============================
# 1. LOAD MODEL
# ============================
model = pickle.load(open("models/model.pkl", "rb"))

# ============================
# 2. DEFINE SAMPLE INPUT
# ============================
# Must match training feature order!
columns = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak",
    "slope","ca","thal"
]

sample_data = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]

df_sample = pd.DataFrame(sample_data, columns=columns)

# ============================
# 3. APPLY SAME SCALING (IMPORTANT)
# ============================
from sklearn.preprocessing import StandardScaler

# NOTE: In real project, save scaler during training!
# Here re-fitting for demonstration only
scaler = StandardScaler()

# These must match scaling columns in training
numeric_cols = ["age","trestbps","chol","thalach","oldpeak"]

df_sample[numeric_cols] = scaler.fit_transform(df_sample[numeric_cols])

# ============================
# 4. PREDICT
# ============================
prediction = model.predict(df_sample)[0]
probability = model.predict_proba(df_sample)[0][1]

# ============================
# 5. OUTPUT
# ============================
print("Prediction:", prediction)
print("Confidence (Probability):", probability)