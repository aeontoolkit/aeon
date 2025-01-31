"""Time series kshapes."""

from typing import Optional, Union

import numpy as np
from numpy.random import RandomState

from aeon.clustering.base import BaseClusterer
from aeon.utils.tags.enum_tags import AlgorithmType


class TimeSeriesKShape(BaseClusterer):
    """Kshape algorithm: wrapper of the ``tslearn`` implementation.

    Parameters
    ----------
    n_clusters: int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
    init: str or np.ndarray, default='random'
        Method for initialising cluster centres. Any of the following are valid:
        ['random']. Or a np.ndarray of shape (n_clusters, n_channels, n_timepoints)
        and gives the initial cluster centres.
    n_init: int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final result will be the best output of n_init
        consecutive runs in terms of inertia.
    max_iter: int, default=30
        Maximum number of iterations of the k-means algorithm for a single
        run.
    tol: float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centres of two consecutive iterations to declare
        convergence.
    verbose: bool, default=False
        Verbosity mode.
    random_state: int or np.random.RandomState instance or None, default=None
        Determines random number generation for centroid initialization.

    Attributes
    ----------
    labels_: np.ndarray (1d array of shape (n_cases,))
        Labels that is the index each time series belongs to.
    inertia_: float
        Sum of squared distances of samples to their closest cluster centre, weighted by
        the sample weights if provided.
    n_iter_: int
        Number of iterations run.

    References
    ----------
    .. [1] John Paparrizos and Luis Gravano. 2016.
       K-Shape: Efficient and Accurate Clustering of Time Series.
       SIGMOD Rec. 45, 1 (March 2016), 69–76.
       https://doi.org/10.1145/2949741.2949758

    Examples
    --------
    >>> from aeon.clustering import TimeSeriesKShape
    >>> from aeon.datasets import load_basic_motions
    >>> # Load data
    >>> X_train, y_train = load_basic_motions(split="TRAIN")[0:10]
    >>> X_test, y_test = load_basic_motions(split="TEST")[0:10]
    >>> # Example of KShapes clustering
    >>> ks = TimeSeriesKShape(n_clusters=3, random_state=1)  # doctest: +SKIP
    >>> ks.fit(X_train)  # doctest: +SKIP
    TimeSeriesKShape(n_clusters=3, random_state=1)
    >>> preds = ks.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        "capability:multivariate": True,
        "python_dependencies": "tslearn",
        "algorithm_type": AlgorithmType.DISTANCE.value,
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init: Union[str, np.ndarray] = "random",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = False,
        random_state: Optional[Union[int, RandomState]] = None,
    ):
        self.n_init = n_init
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.n_clusters = n_clusters

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._tslearn_k_shapes = None

        super().__init__()

    def _fit(self, X, y=None):
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X: np.ndarray, of shape (n_cases, n_channels, n_timepoints) or
                (n_cases, n_timepoints)
            A collection of time series instances.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        self:
            Fitted estimator.
        """
        from tslearn.clustering import KShape

        self._tslearn_k_shapes = KShape(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            n_init=self.n_init,
            verbose=self.verbose,
            init=self.init,
        )

        _X = X.swapaxes(1, 2)

        self._tslearn_k_shapes.fit(_X)
        self._cluster_centers = self._tslearn_k_shapes.cluster_centers_
        self.labels_ = self._tslearn_k_shapes.predict(_X)
        self.inertia_ = self._tslearn_k_shapes.inertia_
        self.n_iter_ = self._tslearn_k_shapes.n_iter_

    def _predict(self, X, y=None) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X: np.ndarray, of shape (n_cases, n_channels, n_timepoints) or
                (n_cases, n_timepoints)
            A collection of time series instances.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_cases,))
            Index of the cluster each time series in X belongs to.
        """
        _X = X.swapaxes(1, 2)
        return self._tslearn_k_shapes.predict(_X)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {
            "n_clusters": 2,
            "init": "random",
            "n_init": 1,
            "max_iter": 1,
            "tol": 1e-4,
            "verbose": False,
            "random_state": 1,
        }
