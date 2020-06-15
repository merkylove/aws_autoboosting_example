import pathlib

import pandas as pd

from autoboosting.auto_estimator import LightGBMWrapper
from autoboosting.constants import ID_COLUMN, TARGET_COLUMN


def train_model(path: pathlib.Path, filename: str, output_path: pathlib.Path,) -> None:

    path_to_train = path / filename

    df = pd.read_csv(path_to_train, index_col=None)

    feature_columns = [i for i in df.columns if i not in (TARGET_COLUMN, ID_COLUMN)]

    model = LightGBMWrapper()
    model.fit(df[feature_columns], df[TARGET_COLUMN])

    output_path = output_path / "classifier"
    print(output_path)

    # save model to output path


if __name__ == "__main__":
    train_model(
        pathlib.Path("/opt/ml/input/data/train"),
        "train.csv",
        pathlib.Path("/opt/ml/model"),
    )
