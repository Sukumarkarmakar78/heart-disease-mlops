import joblib
import pandas as pd

def test_model_prediction():
    model = joblib.load("models/model_pipeline.pkl")

    sample = pd.DataFrame([{
        "age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,
        "fbs":1,"restecg":0,"thalach":150,"exang":0,
        "oldpeak":2.3,"slope":0,"ca":0,"thal":1
    }])

    prediction = model.predict(sample)

    # Should return 1 prediction
    assert len(prediction) == 1
