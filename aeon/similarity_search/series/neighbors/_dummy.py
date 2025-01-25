"""Implementation of NN with brute force."""

from typing import Optional

__maintainer__ = ["baraline"]
__all__ = ["DummySNN"]

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from aeon.similarity_search.series._base import BaseSeriesSimilaritySearch
from aeon.similarity_search.series._commons import (
    _extract_top_k_from_dist_profile,
    _inverse_distance_profile,
)
from aeon.utils.numba.general import (
    get_all_subsequences,
    z_normalise_series_2d,
    z_normalise_series_3d,
)


class DummySNN(BaseSeriesSimilaritySearch):
    """Estimator to compute the on profile and distance profile using brute force."""

    _tags = {"capability:multithreading": True}

    def __init__(
        self,
        length: int,
        normalize: Optional[bool] = False,
        n_jobs: Optional[int] = 1,
    ):
        self.length = length
        self.normalize = normalize
        self.n_jobs = n_jobs
        super().__init__()

    def _fit(
        self,
        X: np.ndarray,
        y=None,
    ):
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)

        self.X_subs = get_all_subsequences(self.X_, self.length, 1)
        if self.normalize:
            self.X_subs = z_normalise_series_3d(self.X_subs)
        set_num_threads(prev_threads)
        return self

    def predict(
        self,
        X: np.ndarray,
        k: Optional[int] = 1,
        dist_threshold: Optional[float] = np.inf,
        exclusion_factor: Optional[float] = 2,
        inverse_distance: Optional[bool] = False,
        allow_neighboring_matches: Optional[bool] = False,
        X_index: Optional[int] = None,
    ):
        """
        Compute nearest neighbors to X in subsequences of X_.

        Parameters
        ----------
        X : np.ndarray, shape=(n_channels, length)
            Subsequence we want to find neighbors for.
        k : int
            The number of neighbors to return.
        dist_threshold : float
            The maximum distance of neighbors to X.
        inverse_distance : bool
            If True, the matching will be made on the inverse of the distance, and thus,
            the farther neighbors will be returned instead of the closest ones.
        exclusion_factor : float, default=1.
            A factor of the query length used to define the exclusion zone when
            ``allow_neighboring_matches`` is set to False. For a given timestamp,
            the exclusion zone starts from
            :math:`id_timestamp - length//exclusion_factor` and end at
            :math:`id_timestamp + length//exclusion_factor`.
        X_index : int, optional
            If ``X`` is a subsequence of X_, specify its starting timestamp in ``X_``.
            If specified, neighboring subsequences of X won't be able to match as
            neighbors.

        Returns
        -------
        np.ndarray, shape = (k)
            The indexes of the best matches in ``distance_profile``.
        np.ndarray, shape = (k)
            The distances of the best matches.

        """
        X = self._pre_predict(X, length=self.length)
        X_index = self._check_X_index(X_index)
        dist_profile = self.compute_distance_profile(X)
        if inverse_distance:
            dist_profile = _inverse_distance_profile(dist_profile)

        exclusion_size = self.length // exclusion_factor
        if X_index is not None:
            exclusion_size = self.length // exclusion_factor
            _max_timestamp = self.n_timepoints_ - self.length
            ub = min(X_index + exclusion_size, _max_timestamp)
            lb = max(0, X_index - exclusion_size)
            dist_profile[lb:ub] = np.inf

        if k == np.inf:
            k = len(dist_profile)

        return _extract_top_k_from_dist_profile(
            dist_profile,
            k,
            dist_threshold,
            allow_neighboring_matches,
            exclusion_size,
        )

    def compute_distance_profile(self, X: np.ndarray):
        """
        Compute the distance profile of X to all samples in X_.

        Parameters
        ----------
        X : np.ndarray, 2D array of shape (n_channels, length)
            The query to use to compute the distance profiles.

        Returns
        -------
        distance_profile : np.ndarray, 1D array of shape (n_candidates)
            The distance profile of X to X_. The ``n_candidates`` value
            is equal to ``n_timepoins - length + 1``, with ``n_timepoints`` the
            length of X_.

        """
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        if self.normalize:
            X = z_normalise_series_2d(X)
        distance_profile = _naive_squared_distance_profile(self.X_subs, X)
        set_num_threads(prev_threads)
        return distance_profile

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        if parameter_set == "default":
            params = {"length": 3}
        else:
            raise NotImplementedError(
                f"The parameter set {parameter_set} is not yet implemented"
            )
        return params


@njit(cache=True, fastmath=True, parallel=True)
def _naive_squared_distance_profile(
    X_subs,
    Q,
):
    """
    Compute a squared euclidean distance profile.

    Parameters
    ----------
    X_subs : array, shape=(n_subsequences, n_channels, length)
        Subsequences of size length of the input time series to search in.
    Q : array, shape=(n_channels, query_length)
        Query used during the search.

    Returns
    -------
    out : np.ndarray, 1D array of shape (n_samples, n_timepoints_t - query_length + 1)
        The distance between the query and all candidates in X.

    """
    n_subs, n_channels, length = X_subs.shape
    dist_profile = np.zeros(n_subs)
    for i in prange(n_subs):
        for j in range(n_channels):
            for k in range(length):
                dist_profile[i] += (X_subs[i, j, k] - Q[j, k]) ** 2
    return dist_profile
