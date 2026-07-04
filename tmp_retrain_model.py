import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib


df = pd.read_csv('data/processed/heart_cleaned.csv')
X = df.drop('target', axis=1)
y = df['target']

numeric_cols = ['age','trestbps','chol','thalach','oldpeak']
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numeric_cols)],
    remainder='passthrough'
)

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model_pipeline.fit(X_train, y_train)
joblib.dump(model_pipeline, 'models/model_pipeline.pkl')
print('saved')
