"""Unit tests for the A3 model (run with: pytest app/code -v)."""
import os
import sys
import pickle

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import model  # noqa: F401,E402  lets pickle find LogisticRegression / CarPriceClassifier

# One realistic raw car row, in the same format the web app sends
SAMPLE = pd.DataFrame([{
    "year": 2015, "km_driven": 60000, "fuel": "Diesel", "seller_type": "Individual",
    "transmission": "Manual", "owner": 1, "mileage": 20.0, "engine": 1248.0,
    "max_power": 74.0, "seats": 5.0, "brand": "Maruti",
}])


@pytest.fixture(scope="module")
def clf():
    with open(os.path.join(BASE_DIR, "model", "a3_model.pkl"), "rb") as f:
        return pickle.load(f)


def test_model_takes_expected_input(clf):
    # After preprocessing: 1 row x (intercept + every feature the model was trained on)
    X = clf.preprocess(SAMPLE)
    assert X.shape == (1, clf.model.n)
    assert not np.isnan(X).any()
    clf.predict(None, SAMPLE)          # must not raise


def test_output_has_expected_shape(clf):
    batch = pd.concat([SAMPLE] * 5, ignore_index=True)
    pred = clf.predict(None, batch)
    assert pred.shape == (5,)                      # one label per input row
    assert set(pred.tolist()) <= {0, 1, 2, 3}      # only valid classes
