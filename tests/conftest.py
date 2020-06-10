import pandas as pd
import pytest

ID_COLUMN = "PassengerId"
TARGET_COLUMN = "Survived"


@pytest.fixture(autouse=True)
def data():
    return pd.read_csv("tests/train.csv")


@pytest.fixture(autouse=True)
def features(data):
    return [c for c in data.columns.tolist() if c not in (ID_COLUMN, TARGET_COLUMN)]
