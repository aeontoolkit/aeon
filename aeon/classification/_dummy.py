"""Dummy time series classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["DummyClassifier"]

import numpy as np
from sklearn.dummy import DummyClassifier as SklearnDummyClassifier

from aeon.classification.base import BaseClassifier


class DummyClassifier(BaseClassifier):
    """
    DummyClassifier makes predictions that ignore the input features.

    This classifier is a wrapper for the scikit-learn DummyClassifier that serves as a
    simple baseline to compare against other more complex classifiers.
    The specific behavior of the baseline is selected with the `strategy` parameter.

    All strategies make predictions that ignore the input feature values passed
    as the `X` argument to `fit` and `predict`. The predictions, however,
    typically depend on values observed in the `y` parameter passed to `fit`.

    Function-identical to `sklearn.dummy.DummyClassifier`, which is called inside.

    Parameters
    ----------
    strategy : {"most_frequent", "prior", "stratified", "uniform", \
            "constant"}, default="prior"
        Strategy to use to generate predictions.
        * "most_frequent": the `predict` method always returns the most
          frequent class label in the observed `y` argument passed to `fit`.
          The `predict_proba` method returns the matching one-hot encoded
          vector.
        * "prior": the `predict` method always returns the most frequent
          class label in the observed `y` argument passed to `fit` (like
          "most_frequent"). ``predict_proba`` always returns the empirical
          class distribution of `y` also known as the empirical class prior
          distribution.
        * "stratified": the `predict_proba` method randomly samples one-hot
          vectors from a multinomial distribution parametrized by the empirical
          class prior probabilities.
          The `predict` method returns the class label which got probability
          one in the one-hot vector of `predict_proba`.
          Each sampled row of both methods is therefore independent and
          identically distributed.
        * "uniform": generates predictions uniformly at random from the list
          of unique classes observed in `y`, i.e. each class has equal
          probability.
        * "constant": always predicts a constant label that is provided by
          the user. This is useful for metrics that evaluate a non-majority
          class.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    constant : int or str or array-like of shape (n_outputs,), default=None
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.

    Examples
    --------
    >>> from aeon.classification.dummy import DummyClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")

    >>> clf = DummyClassifier(strategy="prior")
    >>> clf.fit(X_train, y_train)
    DummyClassifier()
    >>> clf.predict(X_test)[:5]
    array(['1', '1', '1', '1', '1'], dtype='<U1')

    >>> clf = DummyClassifier(strategy="uniform", random_state=0)
    >>> clf.fit(X_train, y_train)
    DummyClassifier(random_state=0, strategy='uniform')
    >>> clf.predict(X_test)[:5]
    array(['1', '2', '2', '1', '2'], dtype='<U1')

    >>> clf = DummyClassifier(strategy="constant", constant="2")
    >>> clf.fit(X_train, y_train)
    DummyClassifier(constant='2', strategy='constant')
    >>> clf.predict(X_test)[:5]
    array(['2', '2', '2', '2', '2'], dtype='<U1')
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:missing_values": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
    }

    def __init__(self, strategy="prior", random_state=None, constant=None):
        self.strategy = strategy
        self.random_state = random_state
        self.constant = constant

        self.sklearn_dummy_classifier = SklearnDummyClassifier(
            strategy=strategy, random_state=random_state, constant=constant
        )

        super().__init__()

    def _fit(self, X, y):
        """Fit the dummy classifier.

        Parameters
        ----------
        X : np.ndarray
            training data shape ``(n_cases, n_channels, n_timepoints)``
        y : np.ndarray
            class labels, shape ``(n_cases)`` - the class labels

        Returns
        -------
        self : reference to self.
        """
        self.sklearn_dummy_classifier.fit(X, y)
        return self

    def _predict(self, X) -> np.ndarray:
        """Predict labels for sequences in X.

        Parameters
        ----------
        X : np.array

        Returns
        -------
        y : predictions of labels for X, np.ndarray
        """
        return self.sklearn_dummy_classifier.predict(X)

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]

        Returns
        -------
        y : predictions of probabilities for class values of X, np.ndarray
        """
        return self.sklearn_dummy_classifier.predict_proba(X)
