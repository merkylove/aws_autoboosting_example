from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from autoboosting.auto_estimator import LightGBMWrapper, ModelNotFittedException
from tests.conftest import TARGET_COLUMN


@pytest.fixture(autouse=True)
def real_fitted_model(data, features):

    """
    This fixture really fits the model and it takes time.
    Comment this and all dependant tests if you don't want to run real training.
    """

    X = data[features]

    model = LightGBMWrapper()
    model.fit(X, data[TARGET_COLUMN])

    return model


def test_custom_lgb_params_applied():

    model = LightGBMWrapper(
        lgb_params={"objective": "binary", "boosting": "gbdt", "metric": "auc"},
        lgb_training_params={"num_boost_round": 50, "early_stopping_rounds": 10},
        metrics={"auc": {"function": roc_auc_score, "use_proba": True}},
    )

    assert model.lgb_params == {
        "objective": "binary",
        "boosting": "gbdt",
        "metric": "auc",
    }

    assert model.lgb_training_params == {
        "num_boost_round": 50,
        "early_stopping_rounds": 10,
    }

    assert model.metrics == {"auc": {"function": roc_auc_score, "use_proba": True}}


def test_wrapper_fit(data, features):

    X = data[features]

    model = LightGBMWrapper()

    # patch real training of lightgbm because it is time consuming
    with patch("lightgbm.train") as m:
        m.return_value = "mocked_value"
        model.fit(X, data[TARGET_COLUMN])

        assert model.categorical_features_extended is not None
        assert model.model == "mocked_value"


def test_predict(real_fitted_model, data, features):

    n = 3

    preds = real_fitted_model.predict(data.iloc[:n][features])

    assert all(preds <= 1)
    assert all(preds >= 0)


def test_predict_proba(real_fitted_model, data, features):

    n = 3

    preds = real_fitted_model.predict_proba(data.iloc[:n][features])[:, 0]

    assert all(preds < 1)
    assert all(preds > 0)


def test_evaluate():

    model = LightGBMWrapper()

    # patch real predictions
    with patch("autoboosting.auto_estimator.LightGBMWrapper.predict") as pr, patch(
        "autoboosting.auto_estimator.LightGBMWrapper.predict_proba"
    ) as pr_pr:

        fake_ys = np.array([1, 0, 0, 1])
        pr.return_value = fake_ys
        pr_pr.return_value = np.array([0.8, 0.2, 0.1, 0.9])

        metrics = model.evaluate(pd.DataFrame(), fake_ys)

    assert metrics["f1_score"] == 1.0


def test_non_fitted_model_exception():
    model = LightGBMWrapper()

    with pytest.raises(ModelNotFittedException):
        model.predict(pd.DataFrame())
