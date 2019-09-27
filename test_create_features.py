from unittest import mock
import pickle

import numpy as np
import pandas as pd

import create_features_python


def mock_load_data():
    return pd.read_csv(
        "./data/test_input_data.csv", encoding="utf-8", dtype=str, na_values="null"
    )


@mock.patch("create_features_python.load_data", mock_load_data)
def test_main():
    with open("./data/test_X_y.pickle", "rb") as f:
        expected_X, expected_y = pickle.load(f)

    X, y = create_features_python.main()
    assert np.array_equal(X, expected_X)
    assert np.array_equal(y, expected_y)
