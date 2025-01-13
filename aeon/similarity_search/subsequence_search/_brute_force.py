"""Implementation of matrix profile with brute force."""

from typing import Optional

__maintainer__ = ["baraline"]


import numpy as np
from numba import njit, prange
from numba.typed import List

from aeon.similarity_search.subsequence_search._base import BaseMatrixProfile
from aeon.similarity_search.subsequence_search._commons import (
    _extract_top_k_from_dist_profile,
    _inverse_distance_profile_list,
)
from aeon.utils.numba.general import (
    get_all_subsequences,
    z_normalise_series_2d,
    z_normalise_series_3d,
)

# TODO : check function params and make docstrings
# TODO : make tests


class BruteForceMatrixProfile(BaseMatrixProfile):
    """Estimator to compute matrix profile and distance profile using brute force."""

    def __init__(
        self,
        length: int,
        normalise: Optional[bool] = False,
        n_jobs: Optional[int] = 1,
    ):
        super().__init__(length=length, n_jobs=n_jobs, normalise=normalise)

    def compute_matrix_profile(
        self,
        k,
        threshold,
        exclusion_size,
        inverse_distance,
        allow_neighboring_matches,
        X: Optional[np.ndarray] = None,
        X_index: Optional[int] = None,
    ):
        """
        Compute matrix profiles.

        The matrix profiles are computed on the collection given in fit. If ``X`` is
        not given, computes the matrix profile of each series in the collection. If it
        is given, only computes it for ``X``.

        Parameters
        ----------
        k : int
            The number of best matches to return during predict for each subsequence.
        threshold : float
            The number of best matches to return during predict for each subsequence.
        inverse_distance : bool
            If True, the matching will be made on the inverse of the distance, and thus,
            the worst matches to the query will be returned instead of the best ones.
        exclusion_size : int
            The size of the exclusion zone used to prevent returning as top k candidates
            the ones that are close to each other (for example i and i+1).
            It is used to define a region between
            :math:`id_timestomp - exclusion_size` and
            :math:`id_timestomp + exclusion_size` which cannot be returned
            as best match if :math:`id_timestomp` was already selected. By default,
            the value None means that this is not used.
        X : Optional[np.ndarray], optional
            The time series on which the matrix profile will be compute.
            The default is None, meaning that the series in the collection given in fit
            will be used instead.
        X_index : Optional[int], optional
            If ``X`` is a series of the database given in fit, specify its index in
            ``X_``. If specified, each query of this series won't be able to match with
            its neighboring subsequences.

        Returns
        -------
        MP : array of shape (series_length - L + 1,)
            Matrix profile distances for each query subsequence. If X is none, this
            will be a list of MP for each series in X_.
        IP : array of shape (series_length - L + 1,)
            Indexes of the top matches for each query subsequence. If X is none, this
            will be a list of MP for each series in X_.
        """
        # pairwise if none
        if X is None:
            MP = []
            IP = []
            for i in range(len(self.X_)):
                _MP, _IP = self.compute_matrix_profile(
                    k,
                    threshold,
                    exclusion_size,
                    inverse_distance,
                    X=self.X_[i],
                    X_index=i,
                )
                MP.append(_MP)
                IP.append(_IP)
        else:
            MP, IP = _naive_squared_matrix_profile(
                self.X_,
                X,
                self.length,
                X_index,
                k,
                threshold,
                allow_neighboring_matches,
                exclusion_size,
                inverse_distance,
                normalise=self.normalise,
            )

        return MP, IP

    def compute_distance_profile(self, X: np.ndarray):
        """
        Compute the distance profile of X to all samples in X_.

        Parameters
        ----------
        X : np.ndarray, 2D array of shape (n_channels, length)
            The query to use to compute the distance profiles.

        Returns
        -------
        distance_profiles : np.ndarray, 2D array of shape (n_cases, n_candidates)
            The distance profile of X to all samples in X_. The ``n_candidates`` value
            is equal to ``n_timepoins - length + 1``. If X_ is an unequal length
            collection, returns a numba typed list instead of an ndarray.

        """
        distance_profiles = _naive_squared_distance_profile(
            self.X_, X, normalise=self.normalise
        )

        if not self.metadata_["unequal_length"]:
            distance_profiles = np.asarray(distance_profiles)
        return distance_profiles


@njit(cache=True, fastmath=True)
def _compute_dist_profile(X_subs, q):
    """
    Compute the distance profile between subsequences and a query.

    Parameters
    ----------
    X_subs : array, shape=(n_samples, n_channels, query_length)
        Input subsequences extracted from a time series.
    q : array, shape=(n_channels, query_length)
        Query used for the distance computation

    Returns
    -------
    dist_profile : np.ndarray, 1D array of shape (n_samples)
        The distance between the query all subsequences.

    """
    n_candidates, n_channels, q_length = X_subs.shape
    dist_profile = np.zeros(n_candidates)
    for i in range(n_candidates):
        for j in range(n_channels):
            for k in range(q_length):
                dist_profile[i] += (X_subs[i, j, k] - q[j, k]) ** 2
    return dist_profile


@njit(cache=True, fastmath=True, parallel=True)
def _naive_squared_distance_profile(
    X,
    Q,
    normalise=False,
):
    """
    Compute a squared euclidean distance profile.

    Parameters
    ----------
    X : array, shape=(n_samples, n_channels, n_timepoints)
        Input time series dataset to search in.
    Q : array, shape=(n_channels, query_length)
        Query used during the search.
    normalise : bool
        Wheter to use a z-normalised distance.

    Returns
    -------
    out : np.ndarray, 1D array of shape (n_samples, n_timepoints_t - query_length + 1)
        The distance between the query and all candidates in X.

    """
    query_length = Q.shape[1]
    dist_profiles = List()
    # Init distance profile array with unequal length support
    for i in range(len(X)):
        dist_profiles.append(np.zeros(X[i].shape[1] - query_length + 1))
    if normalise:
        Q = z_normalise_series_2d(Q)
    else:
        Q = Q.astype(np.float64)

    for _i in prange(len(X)):
        # cast uint64 due to parallel prange
        i = np.int64(_i)
        X_subs = get_all_subsequences(X[i], query_length, 1)
        if normalise:
            X_subs = z_normalise_series_3d(X_subs)

        dist_profile = _compute_dist_profile(X_subs, Q)
        dist_profiles[i] = dist_profile
    return dist_profiles


@njit(cache=True, fastmath=True, parallel=True)
def _naive_squared_matrix_profile(
    X,
    T,
    L,
    T_index,
    k,
    threshold,
    allow_neighboring_matches,
    exclusion_size,
    inverse_distance,
    normalise=False,
):
    """
    Compute a squared euclidean matrix profile.

    Parameters
    ----------
    X:  np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a TypedList
        of 2D arrays of shape (n_channels, n_timepoints)
    T : np.ndarray, 2D array of shape (n_channels, series_length)
        The series used for similarity search. Note that series_length can be equal,
        superior or inferior to n_timepoints, it doesn't matter.
    L : int
        Length of the subsequences used for the distance computation.
    T_index : int,
        If ``T`` is a series of ``X``, specify its index
        in ``X``. If specified, each query of this series won't be able to
        match with its neighboring subsequences.
    k : int
        The number of best matches to return during predict for each subsequence.
    threshold : float
        The number of best matches to return during predict for each subsequence.
    allow_neighboring_matches : bool
        Wheter the top-k candidates can be neighboring subsequences.
    exclusion_size : int
        The size of the exclusion zone used to prevent returning as top k candidates
        the ones that are close to each other (for example i and i+1).
        It is used to define a region between
        :math:`id_timestomp - exclusion_size` and
        :math:`id_timestomp + exclusion_size` which cannot be returned
        as best match if :math:`id_timestomp` was already selected. By default,
        the value None means that this is not used.
    inverse_distance : bool
        If True, the matching will be made on the inverse of the distance, and thus, the
        worst matches to the query will be returned instead of the best ones.
    normalise : bool
        Wheter to use a z-normalised distance.

    Returns
    -------
    out : np.ndarray, 1D array of shape (n_timepoints_t - query_length + 1)
        The minimum distance between each query in T and all candidates in X.
    """
    n_queries = T.shape[1] - L + 1
    MP = List()
    IP = List()

    # Init List to allow parallel, we'll re-use it for all dist profiles
    dist_profiles = List()
    for i_x in range(len(X)):
        dist_profiles.append(np.zeros(X[i_x].shape[1] - L + 1))

    X_subs = List()
    for i in range(len(X)):
        i_subs = get_all_subsequences(X[i], L, 1)
        if normalise:
            i_subs = z_normalise_series_3d(i_subs)
        X_subs.append(i_subs)

    for i_q in range(n_queries):
        Q = T[:, i_q : i_q + L]
        if normalise:
            Q = z_normalise_series_2d(Q)
        for i_x in prange(len(X)):
            dist_profiles[i_x][0 : X[i_x].shape[1] - L + 1] = _compute_dist_profile(
                X_subs[i_x], Q
            )

        if T_index is not None:
            _max_timestamp = X[T_index].shape[1] - L
            ub = min(i_q + exclusion_size, _max_timestamp)
            lb = max(0, i_q - exclusion_size)
            dist_profiles[T_index][lb:ub] = np.inf

        if inverse_distance:
            dist_profiles = _inverse_distance_profile_list(dist_profiles)

        top_indexes, top_dists = _extract_top_k_from_dist_profile(
            dist_profiles,
            k,
            threshold,
            allow_neighboring_matches,
            exclusion_size,
        )

        MP.append(top_dists)
        IP.append(top_indexes)
    return MP, IP
