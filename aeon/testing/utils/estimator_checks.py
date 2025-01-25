"""Utility function for estimator testing."""

__maintainer__ = ["MatthewMiddlehurst"]

import inspect
from inspect import isclass

import numpy as np

from aeon.similarity_search import BaseSimilaritySearch
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.utils.validation import get_n_cases


def _run_estimator_method(estimator, method_name, datatype, split):
    method = getattr(estimator, method_name)
    args = inspect.getfullargspec(method)[0]
    try:
        # forecasting
        if "y" in args and "exog" in args:
            return method(
                y=FULL_TEST_DATA_DICT[datatype][split][0],
                exog=FULL_TEST_DATA_DICT[datatype][split][1],
            )
        # similarity search
        elif "X" in args and "length" in args:
            value = method(
                X=FULL_TEST_DATA_DICT[datatype][split][0],
                length=3,
            )
        # general use
        elif "X" in args and "y" in args:
            value = method(
                X=FULL_TEST_DATA_DICT[datatype][split][0],
                y=FULL_TEST_DATA_DICT[datatype][split][1],
            )
        elif "X" in args:
            value = method(X=FULL_TEST_DATA_DICT[datatype][split][0])
        else:
            value = method()

        # Similarity search return tuple as (distances, indexes)
        if isinstance(estimator, BaseSimilaritySearch):
            if isinstance(value, tuple):
                return value[0]
            else:
                return value
        else:
            return value
    # generic message for ModuleNotFoundError which are assumed to be related to
    # soft dependencies
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Estimator {estimator.__name__} raises a ModuleNotFoundError "
            f"on {method.__name__}. Any required soft dependencies should "
            f'be added to the "python_dependencies" tag, and python version bounds '
            f'should be added to the "python_version" tag.'
        ) from e


def _get_tag(estimator, tag_name, default=None, raise_error=False):
    if estimator is None:
        return None
    elif isclass(estimator):
        return estimator.get_class_tag(
            tag_name=tag_name, raise_error=raise_error, tag_value_default=default
        )
    else:
        return estimator.get_tag(
            tag_name=tag_name, raise_error=raise_error, tag_value_default=default
        )


def _assert_predict_labels(y_pred, datatype, split="test", unique_labels=None):
    if isinstance(datatype, str):
        datatype = FULL_TEST_DATA_DICT[datatype][split][0]

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (get_n_cases(datatype),)
    if unique_labels is not None:
        assert np.all(np.isin(np.unique(y_pred), unique_labels))


def _assert_predict_probabilities(y_proba, datatype, split="test", n_classes=None):
    if isinstance(datatype, str):
        if n_classes is None:
            n_classes = len(np.unique(FULL_TEST_DATA_DICT[datatype][split][1]))
        datatype = FULL_TEST_DATA_DICT[datatype][split][0]

    if n_classes is None:
        raise ValueError(
            "n_classes must be provided if not using a test dataset string"
        )

    assert isinstance(y_proba, np.ndarray)
    assert y_proba.shape == (
        get_n_cases(datatype),
        n_classes,
    )
    assert np.all(y_proba >= 0)
    assert np.all(y_proba <= 1)
    assert np.allclose(np.sum(y_proba, axis=1), 1)
