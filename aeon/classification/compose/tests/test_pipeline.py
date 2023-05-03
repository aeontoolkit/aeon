# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for (dunder) composition functionality attached to the base class."""

__author__ = ["fkiraly"]
__all__ = []

from sklearn.preprocessing import StandardScaler

from aeon.classification.compose import ClassifierPipeline
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.transformations.panel.padder import PaddingTransformer
from aeon.transformations.series.exponent import ExponentTransformer
from aeon.transformations.series.impute import Imputer
from aeon.utils._testing.collection import (
    _make_classification_y,
    _make_nested_dataframe_X,
)
from aeon.utils._testing.estimator_checks import _assert_array_almost_equal


def test_dunder_mul():
    """Test the mul dunder method."""
    RAND_SEED = 42
    y = _make_classification_y(n_instances=10, random_state=RAND_SEED)
    X = _make_nested_dataframe_X(
        n_instances=10, n_timepoints=20, random_state=RAND_SEED, y=y
    )
    X_test = _make_nested_dataframe_X(
        n_instances=5, n_timepoints=20, random_state=RAND_SEED
    )

    t1 = ExponentTransformer(power=4)
    t2 = ExponentTransformer(power=0.25)

    c = KNeighborsTimeSeriesClassifier()
    t12c_1 = t1 * (t2 * c)
    t12c_2 = (t1 * t2) * c
    t12c_3 = t1 * t2 * c

    assert isinstance(t12c_1, ClassifierPipeline)
    assert isinstance(t12c_2, ClassifierPipeline)
    assert isinstance(t12c_3, ClassifierPipeline)

    y_pred = c.fit(X, y).predict(X_test)

    _assert_array_almost_equal(y_pred, t12c_1.fit(X, y).predict(X_test))
    _assert_array_almost_equal(y_pred, t12c_2.fit(X, y).predict(X_test))
    _assert_array_almost_equal(y_pred, t12c_3.fit(X, y).predict(X_test))


def test_mul_sklearn_autoadapt():
    """Test auto-adapter for sklearn in mul."""
    RAND_SEED = 42
    y = _make_classification_y(n_instances=10, random_state=RAND_SEED)
    X = _make_nested_dataframe_X(
        n_instances=10, n_timepoints=20, random_state=RAND_SEED, y=y
    )
    X_test = _make_nested_dataframe_X(
        n_instances=10, n_timepoints=20, random_state=RAND_SEED
    )

    t1 = ExponentTransformer(power=2)
    t2 = StandardScaler()
    c = KNeighborsTimeSeriesClassifier()

    t12c_1 = t1 * (t2 * c)
    t12c_2 = (t1 * t2) * c
    t12c_3 = t1 * t2 * c

    assert isinstance(t12c_1, ClassifierPipeline)
    assert isinstance(t12c_2, ClassifierPipeline)
    assert isinstance(t12c_3, ClassifierPipeline)

    y_pred = t12c_1.fit(X, y).predict(X_test)

    _assert_array_almost_equal(y_pred, t12c_2.fit(X, y).predict(X_test))
    _assert_array_almost_equal(y_pred, t12c_3.fit(X, y).predict(X_test))


def test_missing_unequal_tag_inference():
    """Test that ClassifierPipeline infers missing/unequal tags correctly."""
    c = KNeighborsTimeSeriesClassifier()
    c1 = ExponentTransformer() * PaddingTransformer() * ExponentTransformer() * c
    c2 = ExponentTransformer() * ExponentTransformer() * c
    c3 = Imputer() * ExponentTransformer() * c
    c4 = ExponentTransformer() * Imputer() * c

    assert c1.get_tag("capability:unequal_length")
    assert not c2.get_tag("capability:unequal_length")
    assert c3.get_tag("capability:missing_values")
    assert not c4.get_tag("capability:missing_values")
