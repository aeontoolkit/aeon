"""KSC clusterer implementation."""

from typing import Optional, Union

import numpy as np
from numpy.random import RandomState

from aeon.clustering import TimeSeriesKMeans
from aeon.utils.tags.enum_tags import AlgorithmType


class KSpectralCentroid(TimeSeriesKMeans):
    """K-Spectral Centroid clustering implementation.

    K-Spectral Centroid (k-SC) [1]_ is a clustering algorithm that aims to partition n
    time series using a distance and averaging technique that are scale and shift
    invariant. The algorithm uses an optimisation process to find the best shift and
    scale that minimises the distance between the two time series.

    This version of the algorithm uses the Lloyd's version of the algorithm (follows
    k-means algorithm). There is a second version called "incremental k-SC" that was
    also proposed in the paper. We will look to add this at a later date.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    max_shift : int or None, default=None
        Maximum shift allowed in the alignment path. If None, then max_shift is set
        to min(x.shape[-1], y.shape[-1]). This value is used in both the distance
        and averaging calculations.
    init : str or np.ndarray, default='random'
        Random is the default and simply chooses k time series at random as
        centroids. It is fast but sometimes yields sub-optimal clustering.
        Kmeans++ [2] and is slower but often more
        accurate than random. It works by choosing centroids that are distant
        from one another.
        First is the fastest method and simply chooses the first k time series as
        centroids.
        If a np.ndarray provided it must be of shape (n_clusters, n_channels,
        n_timepoints)
        and contains the time series to use as centroids.
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final result will be the best output of n_init
        consecutive runs in terms of inertia.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single
        run.
    tol : float, default=1e-6
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    verbose : bool, default=False
        Verbosity mode.
    random_state : int, np.random.RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
        If `int`, random_state is the seed used by the random number generator;
        If `np.random.RandomState` instance,
        random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Attributes
    ----------
    cluster_centers_ : 3d np.ndarray
        Array of shape (n_clusters, n_channels, n_timepoints))
        Time series that represent each of the cluster centers.
    labels_ : 1d np.ndarray
        1d array of shape (n_case,)
        Labels that is the index each time series belongs to.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    n_iter_ : int
        Number of iterations run.

    References
    ----------
    .. [1] J. Yang and J. Leskovec. Patterns of temporal variation in online media. In
    Proc. of the fourth ACM international conf. on Web search and data mining, page
    177. ACM, 2011.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.clustering import KSpectralCentroid
    >>> X = np.random.random(size=(10,2,20))
    >>> clst = KSpectralCentroid(n_clusters=2, max_shift=2)
    >>> clst.fit(X)
    KSpectralCentroid(max_shift=2, n_clusters=2)
    >>> preds = clst.predict(X)
    """

    _tags = {
        "capability:multivariate": True,
        "algorithm_type": AlgorithmType.DISTANCE.value,
    }

    def __init__(
        self,
        n_clusters: int = 8,
        max_shift: Optional[int] = None,
        init: Union[str, np.ndarray] = "random",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-6,
        verbose: bool = False,
        random_state: Optional[Union[int, RandomState]] = None,
    ):
        self.max_shift = max_shift

        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            distance="shift_scale",
            averaging_method="shift_scale",
        )

    def _check_params(self, X: np.ndarray) -> None:
        super()._check_params(X)
        temp_max_shift = self.max_shift if self.max_shift is not None else X.shape[-1]
        # set max_shift to the length of the time series
        if "max_shift" not in self._distance_params:
            self._distance_params["max_shift"] = temp_max_shift
        if "max_shift" not in self._average_params:
            self._average_params["max_shift"] = temp_max_shift

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
            "n_init": 1,
            "max_shift": 2,
            "max_iter": 1,
            "random_state": 0,
            "init": "random",
        }
