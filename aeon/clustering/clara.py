# -*- coding: utf-8 -*-
"""Time series kmedoids."""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Callable, Union

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state

from aeon.clustering.base import BaseClusterer
from aeon.clustering.k_medoids import TimeSeriesKMedoids
from aeon.distances import pairwise_distance


class TimeSeriesCLARA(BaseClusterer):
    """Time series CLARA implementation.

    Clustering LARge Applications (CLARA) [1] is a clustering algorithm that
    samples the dataset, applies PAM [2] to the sample, and then uses the
    medoids from the sample to seed PAM on the entire dataset.

    Parameters
    ----------
    n_clusters: int, defaults = 8
        The number of clusters to form as well as the number of
        centroids to generate.
    init_algorithm: str, defaults = 'random'
        Method for initializing cluster centers. Any of the following are valid:
        ['kmedoids++', 'random', 'first'].
    distance: str or Callable, defaults = 'msm'
        Distance metric to compute similarity between time series. Any of the following
        are valid: ['dtw', 'euclidean', 'erp', 'edr', 'lcss', 'squared', 'ddtw', 'wdtw',
        'wddtw', 'msm', 'twe']
    n_samples: int, default = None,
        Number of samples to sample from the dataset. If None, then
        min(n_instances, 40 + 2 * n_clusters) is used.
    n_sampling_iters: int, default = 5,
        Number of different subsets of samples to try. The best subset cluster centres
        are used.
    n_init: int, defaults = 5
        Number of times the PAM algorithm will be run with different
        centroid seeds. The final result will be the best output of n_init
        consecutive runs in terms of inertia.
    max_iter: int, defaults = 300
        Maximum number of iterations of the PAM algorithm for a single
        run.
    tol: float, defaults = 1e-6
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    verbose: bool, defaults = False
        Verbosity mode.
    random_state: int or np.random.RandomState instance or None, defaults = None
        Determines random number generation for centroid initialization.
    distance_params: dict, defaults = None
        Dictionary containing kwargs for the distance metric being used.

    Attributes
    ----------
    cluster_centers_: np.ndarray, of shape (n_instances, n_channels, n_timepoints)
        A collection of time series instances that represent the cluster centres.
    labels_: np.ndarray (1d array of shape (n_instance,))
        Labels that is the index each time series belongs to.
    inertia_: float
        Sum of squared distances of samples to their closest cluster center, weighted by
        the sample weights if provided.
    n_iter_: int
        Number of iterations run.

    Examples
    --------
    >>> from aeon.clustering import TimeSeriesCLARA
    >>> from aeon.datasets import load_basic_motions
    >>> # Load data
    >>> X_train, y_train = load_basic_motions(split="TRAIN")
    >>> X_test, y_test = load_basic_motions(split="TEST")
    >>> # Example of PAM clustering
    >>> km = TimeSeriesCLARA(n_clusters=3, distance="dtw", random_state=1)
    >>> km.fit(X_train)
    TimeSeriesCLARA(distance='dtw', n_clusters=3, random_state=1)
    >>> km.predict(X_test)
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1,
           1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0])

    References
    ----------
    .. [1] Kaufman, Leonard & Rousseeuw, Peter. (1986). Clustering Large Data Sets.
    10.1016/B978-0-444-87877-9.50039-X.
    .. [2] Kaufman, Leonard & Rousseeuw, Peter. (1986). Clustering Large Data Sets.
    10.1016/B978-0-444-87877-9.50039-X.
    """

    _tags = {
        "capability:multivariate": True,
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: Union[str, Callable] = "random",
        distance: Union[str, Callable] = "msm",
        n_samples: int = None,
        n_sampling_iters: int = 10,
        n_init: int = 1,
        max_iter: int = 300,
        tol: float = 1e-6,
        verbose: bool = False,
        random_state: Union[int, RandomState] = None,
        distance_params: dict = None,
    ):
        self.init_algorithm = init_algorithm
        self.distance = distance
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.distance_params = distance_params
        self.n_samples = n_samples
        self.n_sampling_iters = n_sampling_iters

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._random_state = None
        self._init_algorithm = None
        self._distance_cache = None
        self._distance_callable = None
        self._kmedoids_instance = None

        self._distance_params = distance_params
        if distance_params is None:
            self._distance_params = {}

        super(TimeSeriesCLARA, self).__init__(n_clusters)

    def _predict(self, X: np.ndarray, y=None) -> np.ndarray:
        if isinstance(self.distance, str):
            pairwise_matrix = pairwise_distance(
                X, self.cluster_centers_, metric=self.distance, **self._distance_params
            )
        else:
            pairwise_matrix = pairwise_distance(
                X,
                self.cluster_centers_,
                self._distance_callable,
                **self._distance_params,
            )
        return pairwise_matrix.argmin(axis=1)

    def _fit(self, X: np.ndarray, y=None):
        self._random_state = check_random_state(self.random_state)
        n_instances = X.shape[0]
        if self.n_samples is None:
            n_samples = max(
                min(n_instances, 40 + 2 * self.n_clusters), self.n_clusters + 1
            )
        else:
            n_samples = self.n_samples

        best_score = np.inf
        best_pam = None
        for _ in range(self.n_sampling_iters):
            sample_idxs = np.arange(n_samples)
            if n_samples < n_instances:
                sample_idxs = self._random_state.choice(
                    sample_idxs,
                    size=n_samples,
                    replace=False,
                )
            pam = TimeSeriesKMedoids(
                n_clusters=self.n_clusters,
                init_algorithm=self.init_algorithm,
                distance=self.distance,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
                random_state=self._random_state,
                distance_params=self._distance_params,
                method="pam",
            )
            pam.fit(X[sample_idxs])
            if pam.inertia_ < best_score:
                best_pam = pam

        self.labels_ = best_pam.labels_
        self.inertia_ = best_pam.inertia_
        self.cluster_centers_ = best_pam.cluster_centers_
        self.n_iter_ = best_pam.n_iter_

    def _score(self, X, y=None):
        return -self.inertia_

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return {
            "n_clusters": 2,
            "init_algorithm": "random",
            "distance": "euclidean",
            "n_init": 1,
            "max_iter": 1,
            "tol": 0.0001,
            "verbose": False,
            "random_state": 1,
            "n_samples": 10,
            "n_sampling_iters": 5,
        }
