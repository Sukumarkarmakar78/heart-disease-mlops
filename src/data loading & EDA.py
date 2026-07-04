# ============================
# 1. IMPORT LIBRARIES
# ============================
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ============================
# 2. LOAD DATA (LOCAL FILE)
# ============================

# Load file from data folder
df = pd.read_csv("data/raw/heart.csv")

print("Shape:", df.shape)
print(df.head())

# ============================
# 3. DATA CLEANING
# ============================

# Replace '?' with NaN (if present)
df.replace("?", np.nan, inplace=True)

# Convert columns to numeric if needed
df = df.apply(pd.to_numeric, errors='ignore')

# Check missing values
print("\nMissing values:\n", df.isnull().sum())

# Fill missing values (median for numeric)
df.fillna(df.median(numeric_only=True), inplace=True)

# ============================
# 4. TARGET VARIABLE PROCESSING
# ============================

# If target column is named 'target' (most CSV versions)
if "target" in df.columns:
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# ============================
# 5. FEATURE SCALING
# ============================

# Select only numeric columns for scaling
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Remove target from scaling
if "target" in numeric_cols:
    numeric_cols.remove("target")

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ============================
# 6. EXPLORATORY DATA ANALYSIS (EDA)
# ============================

# ---- 6.1 Histograms
df.hist(bins=15, figsize=(15, 10))
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# ---- 6.2 Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ---- 6.3 Class Balance
if "target" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x="target", data=df, palette="viridis")
    plt.title("Class Distribution (Heart Disease)")
    plt.show()

# ---- 6.4 Feature vs Target (if exists)
if "target" in df.columns:
    plt.figure()
    sns.boxplot(x="target", y=df.columns[0], data=df)
    plt.title(f"{df.columns[0]} vs Target")
    plt.show()

# ============================
# 7. SAVE CLEAN DATA
# ============================

os.makedirs("data/processed", exist_ok=True)

df.to_csv("data/processed/heart_cleaned.csv", index=False)

print("✅ Local file loaded + preprocessing + EDA completed!")
