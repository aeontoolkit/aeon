"""Implements DummyClusterer to be used as Baseline."""

import numpy as np
from sklearn.utils import check_random_state

from aeon.clustering.base import BaseClusterer

__all__ = ["DummyClusterer"]


class DummyClusterer(BaseClusterer):
    """
    Dummy clustering for benchmarking purposes.

    This estimator generates cluster labels based on simple strategies
    without considering the input data. It can be used as a baseline
    for evaluating the performance of more complex clustering algorithms.

    Parameters
    ----------
    strategy : str, default="uniform"
        The strategy to use for generating cluster labels. Supported strategies are:
        - "random": Assign clusters randomly.
        - "uniform": Distribute clusters uniformly among samples.
        - "single_cluster": Assign all samples to a single cluster.
    n_clusters : int, default=3
        The number of clusters to generate. This is relevant for "random"
        and "uniform" strategies.
    random_state : int, np.random.RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
        Only used when `strategy` is "random".
        If `int`, random_state is the seed used by the random number generator;
        If `np.random.RandomState` instance,
        random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each sample in the training data.

    Examples
    --------
    >>> from aeon.clustering import DummyClusterer
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> clusterer = DummyClusterer(strategy="uniform", n_clusters=2)
    >>> clusterer.fit(X)
    DummyClusterer(n_clusters=2)
    >>> clusterer.labels_
    array([0, 1, 0])
    >>> clusterer.predict(X)
    array([0, 1, 0])
    """

    def __init__(self, strategy="uniform", n_clusters=3, random_state=None):
        self.strategy = strategy
        self.random_state = random_state
        self.n_clusters = n_clusters

        super().__init__()

    def _fit(self, X, y=None):
        """
        Compute cluster labels for the input data using the specified strategy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        n_samples = X.shape[0]

        if self.strategy == "random":
            rng = check_random_state(self.random_state)
            self.labels_ = rng.randint(self.n_clusters, size=n_samples)
        elif self.strategy == "uniform":
            self.labels_ = np.tile(
                np.arange(self.n_clusters), n_samples // self.n_clusters + 1
            )[:n_samples]
        elif self.strategy == "single_cluster":
            self.labels_ = np.zeros(n_samples, dtype=int)
        else:
            raise ValueError("Unknown strategy type")

        return self

    def _predict(self, X, y=None) -> np.ndarray:
        """
        Predict cluster labels for the input data using the specified strategy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        n_samples = X.shape[0]
        if self.strategy == "random":
            rng = check_random_state(self.random_state)
            return rng.randint(self.n_clusters, size=n_samples)
        elif self.strategy == "uniform":
            return np.tile(
                np.arange(self.n_clusters), n_samples // self.n_clusters + 1
            )[:n_samples]
        elif self.strategy == "single_cluster":
            return np.zeros(n_samples, dtype=int)
        else:
            raise ValueError("Unknown strategy type")
