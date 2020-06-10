from typing import Callable, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split

from autoboosting.preprocessing import RobustLabelEncoder


class ModelNotFittedException(Exception):
    pass


def lgb_compatible_f1_score(
    y_hat: np.ndarray, data: lgb.Dataset
) -> Tuple[str, float, bool]:
    y_true = data.get_label().astype(int)
    y_hat = np.round(y_hat).astype(int)  # scikit's f1 doesn't work with probabilities

    return "f1_score", f1_score(y_true, y_hat), True


class LightGBMWrapper:
    def __init__(
        self,
        lgb_params: Optional[Dict] = None,
        lgb_training_params: Optional[Dict] = None,
        categorical_features: Optional[List] = None,
        random_state: Optional[int] = None,
        metrics: Optional[Dict[str, Dict[str, Union[bool, Callable]]]] = None,
    ):
        """
        This class deals with categorical features instead of you, fits label encoders,
        handles unseen values for categories, fits the model,
        performs predictions and evaluation
        :param lgb_params:
        :param lgb_training_params:
        :param categorical_features:
        :param random_state:
        :param metrics:
        """

        if not lgb_params:
            self.lgb_params = {
                "objective": "binary",
                "boosting": "gbdt",
                "metric": "f1_score",
                "num_leaves": 7,
                "feature_fraction": 0.5,
                "bagging_fraction": 0.5,
                "learning_rate": 0.01,
                "verbose": -1,
                "nthreads": -1,
                "bagging_seed": random_state,
                "seed": random_state,
                "feature_fraction_seed": random_state,
            }
        else:
            self.lgb_params = lgb_params

        self.label_encoders: Dict[str, RobustLabelEncoder] = {}
        self.categorical_features = categorical_features
        self.categorical_features_extended: Union[None, Tuple[str, ...]] = None
        self.columns: Union[None, Tuple[str, ...]] = None

        if not lgb_training_params:
            self.lgb_training_params = {
                "num_boost_round": 5000,
                "early_stopping_rounds": 200,
                "feval": lgb_compatible_f1_score,
            }
        else:
            self.lgb_training_params = lgb_training_params

        self.model = None

        if metrics is None:
            self.metrics = {
                "f1_score": {"function": f1_score, "use_proba": False},
                "logloss": {"function": log_loss, "use_proba": True},
            }
        else:
            self.metrics = metrics

        # for multiclass case we assume the same threshold for each class
        # in general case it's better to tune threshold w.r.t. class
        # TODO: add threshold as parameters to constructor
        self.default_threshold = 0.5
        self.random_state = random_state

    def reset_state(self):
        self.label_encoders = {}
        self.categorical_features_extended = None
        self.columns = None
        self.model = None

    def fit(self, X: pd.DataFrame, y: np.array) -> None:
        """
        Note: according to sklearn conventions .fit returns self but in
        this takehome they require to return None
        """

        self.reset_state()

        # by default we assume that we shouldn't change input data
        # but it increases RAM consumption
        # TODO: add parameter like 'inplace' to allow user to reduce memory usage
        X = X.copy(deep=True)

        columns = X.columns.tolist()

        new_categorical_features = []

        for column in columns:
            if X[column].dtype == "O" or (
                self.categorical_features and column in self.categorical_features
            ):

                unseen_value = "<UNSEEN_VALUE>"
                if X[column].dtype != "O":
                    unseen_value = np.inf

                label_encoder = RobustLabelEncoder(unseen_value=unseen_value).fit(
                    X[column].values
                )
                self.label_encoders[column] = label_encoder
                X[column] = label_encoder.transform(X[column])

                new_categorical_features.append(column)

        if self.categorical_features:
            self.categorical_features_extended = tuple(
                sorted(list(set(self.categorical_features + new_categorical_features)))
            )
        else:
            self.categorical_features_extended = tuple(sorted(new_categorical_features))

        self.columns = tuple(sorted(X.columns.tolist()))

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=0.8, stratify=y, random_state=self.random_state
        )

        train_dataset = lgb.Dataset(
            X_train[list(self.columns)],
            y_train,
            params={"verbose": -1},
            free_raw_data=False,
            categorical_feature=self.categorical_features_extended,
        )
        val_dataset = lgb.Dataset(
            X_val[list(self.columns)],
            y_val,
            params={"verbose": -1},
            free_raw_data=False,
            categorical_feature=self.categorical_features_extended,
        )

        self.model = lgb.train(
            self.lgb_params,
            train_dataset,
            valid_sets=(train_dataset, val_dataset),
            valid_names=("train", "val"),
            categorical_feature=self.categorical_features_extended,
            verbose_eval=False,
            **self.lgb_training_params,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:

        X = X.copy(deep=True)

        if self.categorical_features_extended:
            for column in self.categorical_features_extended:
                X[column] = self.label_encoders[column].transform(X[column])

        if self.model is None:
            raise ModelNotFittedException
        y_hat = self.model.predict(X[list(self.columns)])

        # for binary classification
        # cast to format expected by Datadog requirement
        # (refer to p.3, example output)
        if len(y_hat.shape) == 1:
            y_hat = np.array([y_hat, 1.0 - y_hat]).T

        return y_hat

    def predict(self, X: pd.DataFrame) -> np.ndarray:

        proba = self.predict_proba(X)

        # for binary classification
        if proba.shape[1] == 2:
            proba = proba[:, 0] > self.default_threshold
        else:
            proba = proba > self.default_threshold

        return proba.astype("int")

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:

        # to avoid making predictions twice we can repeat logic from .predict here
        # but do not care about performance right now
        proba = self.predict_proba(X)
        classes = self.predict(X)

        metrics = {}
        for metric_name, params in self.metrics.items():

            if params["use_proba"]:
                y_pred = proba
            else:
                y_pred = classes
            metrics[metric_name] = params["function"](y, y_pred)

        return metrics
