"""Tests for ProximityForest2.0."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from aeon.classification.distance_based import ProximityForest2, ProximityTree2
from aeon.datasets import load_unit_test


def test_get_parameter_value():
    """Test the distance parameters generated."""
    X = np.random.rand(10, 50)
    random_state = 42
    tree = ProximityTree2(random_state=random_state)

    params = tree._get_parameter_value(X)

    # Check if the parameters are generated for all distance measures
    expected_measures = [
        "dtw",
        "adtw",
        "lcss",
    ]
    assert set(params.keys()) == set(expected_measures)

    # Check specific parameter ranges
    for measure, measure_params in params.items():
        if measure in ["dtw", "adtw", "lcss"]:
            assert 0 <= measure_params["window"] <= 0.25
        elif measure == "lcss":
            X_std = X.std()
            assert X_std / 5 <= measure_params["epsilon"] <= X_std


def test_get_cadidate_splitter():
    """Test the method to generate candidate splitters."""
    X, y = load_unit_test()
    clf = ProximityTree2(max_depth=1)
    clf.fit(X, y)
    splitter = clf._get_candidate_splitter(X, y)
    assert len(splitter) == 3

    expected_measures = [
        "dtw",
        "adtw",
        "lcss",
    ]
    measure = list(splitter[1].keys())[0]
    assert measure in expected_measures


def test_get_best_splitter():
    """Test the method to get optimum splitter of a node."""
    X, y = load_unit_test()
    clf = ProximityTree2(n_splitters=3, max_depth=1)
    clf.fit(X, y)

    splitter = clf._get_best_splitter(X, y)

    assert splitter is not None

    assert isinstance(splitter, list)

    assert len(splitter) == 3


def test_proximity_tree():
    """Test the fit method of ProximityTree."""
    X, y = load_unit_test()
    clf = ProximityTree2(n_splitters=3, max_depth=4)
    clf.fit(X, y)
    X_test, y_test = load_unit_test(split="train")
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    assert score >= 0.9


def test_univariate():
    """Test that the function gives appropriate error message."""
    X, y = load_unit_test()
    X_multivariate = X.reshape((-1, 2, 12))
    clf = ProximityForest2(n_trees=5)
    with pytest.raises(ValueError):
        clf.fit(X_multivariate, y)


def test_proximity_forest():
    """Test the fit method of ProximityTree."""
    X_train, y_train = load_unit_test()
    X_test, y_test = load_unit_test(split="test")
    clf = ProximityForest2(n_trees=5, n_splitters=3, max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    assert score >= 0.9
