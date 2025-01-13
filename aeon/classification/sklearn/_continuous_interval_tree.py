"""Continuous interval tree (CIT) vector classifier (aka Time Series Tree).

Continuous Interval Tree aka Time Series Tree, base classifier originally used
in the time series forest interval based classification algorithm. Fits sklearn
conventions.
"""

__maintainer__ = []
__all__ = ["ContinuousIntervalTree"]

import math
import sys
from typing import Union

import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import issparse
from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets


class _TreeNode:
    """ContinuousIntervalTree tree node."""

    def __init__(
        self, random_state: Union[int, np.random.RandomState, None] = None
    ) -> None:
        self.random_state = random_state

        self.best_split = -1
        self.best_threshold = 0
        self.best_gain = 0.000001
        self.best_margin = -1
        self.children = []
        self.leaf_distribution = []
        self.depth = -1

    def build_tree(
        self,
        X,
        y,
        thresholds: int,
        entropy: float,
        distribution,
        depth: int,
        max_depth: int,
        n_classes: int,
        leaf: bool,
    ):
        self.depth = depth
        best_distributions = []
        best_entropies = []

        if leaf is False and self.remaining_classes(distribution) and depth < max_depth:
            for (_, att), threshold in np.ndenumerate(thresholds):
                (
                    info_gain,
                    distributions,
                    entropies,
                ) = self.information_gain(X, y, att, threshold, entropy, n_classes)

                if info_gain > self.best_gain:
                    self.best_split = att
                    self.best_threshold = threshold
                    self.best_gain = info_gain
                    self.best_margin = -1
                    best_distributions = distributions
                    best_entropies = entropies
                elif info_gain == self.best_gain and info_gain > 0.000001:
                    margin = self.margin_gain(X, att, threshold)
                    if self.best_margin == -1:
                        self.best_margin = self.margin_gain(
                            X, self.best_split, self.best_threshold
                        )

                    if margin > self.best_margin or (
                        margin == self.best_margin
                        and self.random_state.choice([True, False])
                    ):
                        self.best_split = att
                        self.best_threshold = threshold
                        self.best_margin = margin
                        best_distributions = distributions
                        best_entropies = entropies

        if self.best_split > -1:
            self.children = [None, None, None]

            left_idx, right_idx, missing_idx = self.split_data(
                X, self.best_split, self.best_threshold
            )

            if len(left_idx) > 0:
                self.children[0] = _TreeNode(random_state=self.random_state)
                self.children[0].build_tree(
                    X[left_idx],
                    y[left_idx],
                    thresholds,
                    best_entropies[0],
                    best_distributions[0],
                    depth + 1,
                    max_depth,
                    n_classes,
                    False,
                )
            else:
                self.children[0] = _TreeNode(random_state=self.random_state)
                self.children[0].build_tree(
                    X,
                    y,
                    thresholds,
                    entropy,
                    distribution,
                    depth + 1,
                    max_depth,
                    n_classes,
                    True,
                )

            if len(right_idx) > 0:
                self.children[1] = _TreeNode(random_state=self.random_state)
                self.children[1].build_tree(
                    X[right_idx],
                    y[right_idx],
                    thresholds,
                    best_entropies[1],
                    best_distributions[1],
                    depth + 1,
                    max_depth,
                    n_classes,
                    False,
                )
            else:
                self.children[1] = _TreeNode(random_state=self.random_state)
                self.children[1].build_tree(
                    X,
                    y,
                    thresholds,
                    entropy,
                    distribution,
                    depth + 1,
                    max_depth,
                    n_classes,
                    True,
                )

            if len(missing_idx) > 0:
                self.children[2] = _TreeNode(random_state=self.random_state)
                self.children[2].build_tree(
                    X[missing_idx],
                    y[missing_idx],
                    thresholds,
                    best_entropies[2],
                    best_distributions[2],
                    depth + 1,
                    max_depth,
                    n_classes,
                    False,
                )
            else:
                self.children[2] = _TreeNode(random_state=self.random_state)
                self.children[2].build_tree(
                    X,
                    y,
                    thresholds,
                    entropy,
                    distribution,
                    depth + 1,
                    max_depth,
                    n_classes,
                    True,
                )
        else:
            self.leaf_distribution = distribution / np.sum(distribution)

        return self

    def predict_proba(self, X, n_classes: int):
        if self.best_split > -1:
            if X[self.best_split] <= self.best_threshold:
                return self.children[0].predict_proba(X, n_classes)
            elif X[self.best_split] > self.best_threshold:
                return self.children[1].predict_proba(X, n_classes)
            else:
                return self.children[2].predict_proba(X, n_classes)
        else:
            return self.leaf_distribution

    @staticmethod
    @njit(fastmath=True, cache=True)
    def information_gain(
        X, y, attribute: int, threshold: int, parent_entropy: float, n_classes: int
    ):
        dist_left = np.zeros(n_classes)
        dist_right = np.zeros(n_classes)
        dist_missing = np.zeros(n_classes)
        for i, case in enumerate(X):
            if case[attribute] <= threshold:
                dist_left[y[i]] += 1
            elif case[attribute] > threshold:
                dist_right[y[i]] += 1
            else:
                dist_missing[y[i]] += 1

        sum_missing = 0
        for v in dist_missing:
            sum_missing += v
        sum_left = 0
        for v in dist_left:
            sum_left += v
        sum_right = 0
        for v in dist_right:
            sum_right += v

        entropy_left = _entropy(dist_left, sum_left)
        entropy_right = _entropy(dist_right, sum_right)
        entropy_missing = _entropy(dist_missing, sum_missing)

        n_cases = X.shape[0]
        info_gain = (
            parent_entropy
            - sum_left / n_cases * entropy_left
            - sum_right / n_cases * entropy_right
            - sum_missing / n_cases * entropy_missing
        )

        return (
            info_gain,
            [dist_left, dist_right, dist_missing],
            [entropy_left, entropy_right, entropy_missing],
        )

    @staticmethod
    @njit(fastmath=True, cache=True)
    def margin_gain(X, attribute: int, threshold: int):
        margins = np.abs(X[:, attribute] - threshold)
        return np.min(margins)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def split_data(X, best_split: int, best_threshold: int):
        left_idx = np.zeros(len(X), dtype=np.int_)
        left_count = 0
        right_idx = np.zeros(len(X), dtype=np.int_)
        right_count = 0
        missing_idx = np.zeros(len(X), dtype=np.int_)
        missing_count = 0
        for i, case in enumerate(X):
            if case[best_split] <= best_threshold:
                left_idx[left_count] = i
                left_count += 1
            elif case[best_split] > best_threshold:
                right_idx[right_count] = i
                right_count += 1
            else:
                missing_idx[missing_count] = i
                missing_count += 1

        return (
            left_idx[:left_count],
            right_idx[:right_count],
            missing_idx[:missing_count],
        )

    @staticmethod
    @njit(fastmath=True, cache=True)
    def remaining_classes(distribution) -> bool:
        remaining_classes = 0
        for d in distribution:
            if d > 0:
                remaining_classes += 1
        return remaining_classes > 1


class ContinuousIntervalTree(ClassifierMixin, BaseEstimator):
    """Continuous interval tree (CIT) vector classifier (aka Time Series Tree).

    The `Time Series Tree` described in the Time Series Forest (TSF) [1]_. A simple
    information gain based tree for continuous attributes using a bespoke margin gain
    metric for tie breaking.

    Implemented as a bade classifier for interval based time series classifiers such as
    `CanonicalIntervalForest` and `DrCIF`.

    Parameters
    ----------
    max_depth : int, default=sys.maxsize
        Maximum depth for the tree.
    thresholds : int, default=20
        Number of thresholds to split continous attributes on at tree nodes.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Attributes
    ----------
    classes_ : list
        The unique class labels in the training set.
    n_classes_ : int
        The number of unique classes in the training set.
    n_cases_ : int
        The number of train cases in the training set.
    n_atts_ : int
        The number of attributes in the training set.

    See Also
    --------
    CanonicalIntervalForest
    DrCIF

    Notes
    -----
    For the Java version, see
    `tsml <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    machine_learning/classifiers/ContinuousIntervalTree.java>`_.

    References
    ----------
    .. [1] H.Deng, G.Runger, E.Tuv and M.Vladimir, "A time series forest for
       classification and feature extraction",Information Sciences, 239, 2013

    Examples
    --------
    >>> from aeon.classification.sklearn import ContinuousIntervalTree
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = ContinuousIntervalTree()
    >>> clf.fit(X_train, y_train)
    ContinuousIntervalTree(...)
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        max_depth: int = sys.maxsize,
        thresholds: int = 20,
        random_state: Union[int, np.random.RandomState, None] = None,
    ) -> None:
        self.max_depth = max_depth
        self.thresholds = thresholds
        self.random_state = random_state

        super().__init__()

    def fit(self, X, y):
        """Fit a tree on cases (X,y), where y is the target variable.

        Build an information gain based tree for continuous attributes using the
        margin gain metric for ties.

        Parameters
        ----------
        X : 2d ndarray or DataFrame of shape = [n_cases, n_attributes]
            The training data.
        y : array-like, shape = [n_cases]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_".
        """
        # data processing
        X = self._check_X(X)
        X, y = self._validate_data(
            X=X,
            y=y,
            ensure_min_samples=2,
            force_all_finite="allow-nan",
            accept_sparse=False,
        )
        check_classification_targets(y)

        self.n_cases_, self.n_atts_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self._class_dictionary = {}
        for index, classVal in enumerate(self.classes_):
            self._class_dictionary[classVal] = index

        # escape if only one class seen
        if self.n_classes_ == 1:
            self._is_fitted = True
            return self

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        rng = check_random_state(self.random_state)
        self._root = _TreeNode(random_state=rng)

        thresholds = np.linspace(np.min(X, axis=0), np.max(X, axis=0), self.thresholds)

        _, distribution = np.unique(y, return_counts=True)

        entropy = _entropy(distribution, distribution.sum())

        self._root.build_tree(
            X,
            y,
            thresholds,
            entropy,
            distribution,
            0,
            self.max_depth,
            self.n_classes_,
            False,
        )

        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict for all cases in X. Built on top of predict_proba.

        Parameters
        ----------
        X : 2d ndarray or DataFrame of shape = [n_cases, n_attributes]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted class labels.
        """
        return np.array(
            [self.classes_[int(np.argmax(prob))] for prob in self.predict_proba(X)]
        )

    def predict_proba(self, X):
        """Probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : 2d ndarray or DataFrame of shape = [n_cases, n_attributes]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_cases, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        # data processing
        X = self._check_X(X)
        X = self._validate_data(
            X=X, reset=False, force_all_finite="allow-nan", accept_sparse=False
        )

        dists = np.zeros((X.shape[0], self.n_classes_))
        for i in range(X.shape[0]):
            dists[i] = self._root.predict_proba(X[i], self.n_classes_)
        return dists

    def tree_node_splits_and_gain(self) -> tuple[list[int], list[float]]:
        """Recursively find the split and information gain for each tree node."""
        splits = []
        gains = []

        if self._root.best_split > -1:
            self._find_splits_gain(self._root, splits, gains)

        return splits, gains

    def _find_splits_gain(self, node: _TreeNode, splits: list, gains: list):
        """Recursively find the split and information gain for each tree node."""
        splits.append(node.best_split)
        gains.append(node.best_gain)

        for next_node in node.children:
            if next_node.best_split > -1:
                self._find_splits_gain(next_node, splits, gains)

    def _check_X(self, X):
        if issparse(X):
            return X

        msg = (
            "ContinuousIntervalTree is not a time series classifier. "
            "A valid sklearn input such as a 2d numpy array is required."
            "Sparse input formats are currently not supported."
        )
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        else:
            try:
                X = np.array(X)
            except Exception:
                raise ValueError(msg)

        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(msg)

        return X


@njit(fastmath=True, cache=True)
def _entropy(x, s: int) -> float:
    e = 0
    for i in x:
        p = i / s if s > 0 else 0
        e += -(p * math.log(p) / 0.6931471805599453) if p > 0 else 0
    return e
