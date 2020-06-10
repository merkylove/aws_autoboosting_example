import numpy as np
import pytest

from autoboosting.preprocessing import RobustLabelEncoder


@pytest.mark.parametrize(
    "fit_labels, transform_labels, expected_output, unseen_value_placeholder",
    (
        (("A", "B", "C"), ("C", "B", "A"), (3, 2, 1), "<UNKNOWN>"),
        (("A", "B", "C"), ("C", "B", "A", "D"), (3, 2, 1, 0), "<UNKNOWN>"),
        (
            ("A", "B", "C", "C", "C"),
            ("C", "B", "A", "D", "E"),
            (3, 2, 1, 0, 0),
            "<UNKNOWN>",
        ),
        (
            ("A", "B", "C"),
            ("C", "B", "A", "D", "E", None),
            (3, 2, 1, 0, 0, 0),
            "<UNKNOWN>",
        ),
        ((0, 1, 2), (2, 1, 0, 123, 555, None), (3, 2, 1, 0, 0, 0), -1),
        (("",), ("C", "B", "A"), (1, 1, 1), "<UNKNOWN>"),
        (("A", "B", "C"), tuple(), tuple(), "<UNKNOWN>"),
        (
            np.array([1, 2, 37, 9]),
            np.array([1, 1, 1, 2, 2, 2, 23, 9, 9, 3]),
            (1, 1, 1, 2, 2, 2, 0, 3, 3, 0),
            -1,
        ),
    ),
)
def test_label_encoding(
    fit_labels, transform_labels, expected_output, unseen_value_placeholder
):

    encoder = RobustLabelEncoder(unseen_value_placeholder)
    encoder.fit(fit_labels)
    output = encoder.transform(transform_labels)

    assert all(output == expected_output)
