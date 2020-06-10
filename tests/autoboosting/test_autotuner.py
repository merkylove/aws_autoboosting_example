from unittest.mock import patch

import numpy as np

from autoboosting.autotuner import (
    AutoTuningLightGBMWrapper,
    lightgbm_wrapper_cross_val_score,
)
from autoboosting.settings import RANDOM_SEED
from tests.conftest import TARGET_COLUMN


def test_lgb_wrapper_cv(data, features):

    with patch("autoboosting.auto_estimator.LightGBMWrapper.fit") as fit, patch(
        "autoboosting.auto_estimator.LightGBMWrapper.evaluate"
    ) as evaluate:

        fit.return_value = None
        evaluate.return_value = {"f1_score": 0.9, "logloss": 1.0}

        metrics = lightgbm_wrapper_cross_val_score(
            {}, data[features], data[TARGET_COLUMN]
        )

        assert metrics["f1_score"] == 0.9


def test_tuner(data, features):

    with patch("autoboosting.autotuner.lightgbm_wrapper_cross_val_score") as cv:

        cv.return_value = {"f1_score": 0.9, "logloss": 1.0}

        at = AutoTuningLightGBMWrapper(
            hyperparameters_tuning_params={"init_points": 1, "n_iter": 2}
        )
        best_params = at.tune_parameters(
            data.iloc[:10][features], data.iloc[:10][TARGET_COLUMN]
        )

        assert isinstance(best_params, dict)


def test_reproducibility(data, features):

    """
    This test takes much time to run because
    it honestly runs fitting and parameters tuning on a small dataset.
    If you're not patient enough, you can comment it and
    do not rely on complete reproducibility
    """

    np.random.seed(RANDOM_SEED)
    at = AutoTuningLightGBMWrapper()
    results1 = at.tune_parameters(data[features], data[TARGET_COLUMN])

    np.random.seed(RANDOM_SEED)
    at = AutoTuningLightGBMWrapper()
    results2 = at.tune_parameters(data[features], data[TARGET_COLUMN])

    assert results1 == results2
