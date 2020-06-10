import copy
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold

from autoboosting.auto_estimator import LightGBMWrapper
from autoboosting.settings import RANDOM_SEED


def lightgbm_wrapper_cross_val_score(
    lgb_wrapper_params,
    X: pd.DataFrame,
    y: np.ndarray,
    random_state: Optional[int] = None,
    cv: int = 2,
):
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    metrics_agg: Dict[str, List[float]] = defaultdict(list)

    for train, test in kfold.split(X, y):

        estimator = LightGBMWrapper(**lgb_wrapper_params)

        estimator.fit(X.iloc[train], y[train])
        metrics = estimator.evaluate(X.iloc[test], y[test])

        for k, v in metrics.items():
            metrics_agg[k].append(v)

    return {k: np.mean(v) for k, v in metrics_agg.items()}


class AutoTuningLightGBMWrapper(LightGBMWrapper):
    def __init__(
        self,
        lgb_params: Optional[Dict] = None,
        lgb_training_params: Optional[Dict] = None,
        categorical_features: Optional[List] = None,
        metrics: Optional[Dict[str, Dict[str, Union[bool, Callable]]]] = None,
        parameters_to_tune: Dict = None,
        hyperparameters_tuning_params: Dict = None,
        metric_to_tune: str = "f1_score",
        random_state: Optional[int] = RANDOM_SEED,
    ):
        """
        This class provides opportunity to find best parameters to optimize your metric.
        :param lgb_params: refer to lightgbm documentation to find all options
        :param lgb_training_params:
        :param categorical_features:
        :param metrics: metrics format
            {
                "METRIC_NAME": {
                    "function": how to calculate your metric (y_true, y_pred, ...)
                    "use_proba": pass classes or probabilities to the function
                    }
            }
        :param parameters_to_tune:
            specify desired parameters here,
            refer to Bayesian Optimization package for details
        :param hyperparameters_tuning_params:
            refer to Bayesian Optimization package for details
        :param metric_to_tune:
            make sure that metric you want to tune is described in metrics parameter
            (by default, only 'f1_score' and 'logloss' are supported)
        :param random_state:
        """

        super(AutoTuningLightGBMWrapper, self).__init__(
            lgb_params=lgb_params,
            lgb_training_params=lgb_training_params,
            categorical_features=categorical_features,
            metrics=metrics,
            random_state=random_state,
        )

        if not parameters_to_tune:
            # just an example, there are many other reasonable parameters to tune
            self.parameters_to_tune = {
                "num_leaves": (3, 15),
                "feature_fraction": (0.05, 0.95),
                "bagging_fraction": (0.05, 0.95),
                "learning_rate": (0.05, 0.1),
                "max_bin": (5, 20),
                "min_data_in_leaf": (5, 10),
            }
        else:
            self.parameters_to_tune = parameters_to_tune

        if hyperparameters_tuning_params:
            self.hyperparameters_tuning_params = hyperparameters_tuning_params
        else:
            self.hyperparameters_tuning_params = {"init_points": 2, "n_iter": 5}

        self.random_state = random_state
        self.metric_to_tune = metric_to_tune

    def cast_params(
        self, params: Dict[str, Union[float, int]]
    ) -> Dict[str, Union[float, int]]:

        casted_params: Dict[str, Union[float, int]] = {}

        for param, value in params.items():

            param_lower_bound = self.parameters_to_tune[param][0]

            if isinstance(param_lower_bound, int):
                casted_params[param] = int(round(value))
            else:
                casted_params[param] = value

        return casted_params

    def tune_parameters(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        def wrapper(**kwargs):

            params = copy.deepcopy(kwargs)

            params = self.cast_params(params)

            if self.lgb_params:
                lgb_params = copy.deepcopy(self.lgb_params)
                lgb_params.update(params)
            else:
                lgb_params = params

            return lightgbm_wrapper_cross_val_score(
                {
                    "lgb_params": lgb_params,
                    "lgb_training_params": self.lgb_training_params,
                    "categorical_features": self.categorical_features,
                },
                X,
                y,
                random_state=self.random_state,
            )[self.metric_to_tune]

        tuner = BayesianOptimization(
            f=wrapper,
            pbounds=self.parameters_to_tune,
            random_state=self.random_state,
            verbose=0,
        )

        tuner.maximize(**self.hyperparameters_tuning_params)

        return {
            "best_parameters": self.cast_params(tuner.max["params"]),
            "best_score": tuner.max["target"],
        }
