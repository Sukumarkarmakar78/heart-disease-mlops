import pandas as pd


def test_data_not_empty():
    df = pd.read_csv("data/raw/heart.csv")
    assert len(df) > 0
