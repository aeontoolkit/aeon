"""Implementation of STOMP with squared euclidean distance."""

__maintainer__ = ["baraline"]
from typing import Optional

import numpy as np
from numba import njit, prange
from numba.typed import List

from aeon.similarity_search.subsequence_search._base import BaseMatrixProfile
from aeon.similarity_search.subsequence_search._commons import (
    _extract_top_k_from_dist_profile,
    _inverse_distance_profile_list,
    fft_sliding_dot_product,
    get_ith_products,
)
from aeon.utils.numba.general import (
    AEON_NUMBA_STD_THRESHOLD,
    sliding_mean_std_one_series,
)


class StompMatrixProfile(BaseMatrixProfile):
    """Estimator to compute matrix profile and distance profile using STOMP."""

    def __init__(
        self,
        length: int,
        normalise: Optional[bool] = False,
        n_jobs: Optional[int] = 1,
    ):
        super().__init__(length=length, n_jobs=n_jobs, normalise=normalise)

    def compute_matrix_profile(
        self,
        X: np.ndarray,
        k: int,
        threshold: float,
        exclusion_size: int,
        inverse_distance: bool,
        allow_neighboring_matches: bool,
        X_index=None,
    ):
        """
        Compute matrix profiles.

        The matrix profiles are computed on the collection given in fit. If ``X`` is
        not given, computes the matrix profile of each series in the collection. If it
        is given, only computes it for ``X``.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_timepoints)
            A 2D array time series on which the matrix profile will be computed.
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
            :math:`id_timestamp - exclusion_size` and
            :math:`id_timestamp + exclusion_size` which cannot be returned
            as best match if :math:`id_timestamp` was already selected. By default,
            the value None means that this is not used.
        X_index : Optional[int], optional
            If ``X`` is a series of the database given in fit, specify its index in
            ``X_``. If specified, each query of this series won't be able to match with
            its neighboring subsequences.

        Returns
        -------
        MP : array of shape (n_timepoints - L + 1,)
            Matrix profile distances for each query subsequence. If X is none, this
            will be a list of MP for each series in X_.
        IP : array of shape (n_timepoints - L + 1,)
            Indexes of the top matches for each query subsequence. If X is none, this
            will be a list of MP for each series in X_.
        """
        XdotT = [
            get_ith_products(self.X[i], X, self.length, 0) for i in range(len(self.X_))
        ]
        if isinstance(X, np.ndarray):
            XdotT = np.asarray(XdotT)
        elif isinstance(X, List):
            XdotT = List(XdotT)

        if X_index is None:
            X_means, X_stds = sliding_mean_std_one_series(X, self.length, 1)
        else:
            X_means, X_stds = self.X_means_[X_index], self.X_stds_[X_index]
        if self.normalise:
            MP, IP = _stomp_normalised(
                self.X_,
                X,
                XdotT,
                self.X_means_,
                self.X_stds_,
                X_means,
                X_stds,
                self.length,
                X_index,
                k,
                threshold,
                allow_neighboring_matches,
                exclusion_size,
                inverse_distance,
            )
        else:
            MP, IP = _stomp(
                self.X_,
                X,
                XdotT,
                self.length,
                X_index,
                k,
                threshold,
                allow_neighboring_matches,
                exclusion_size,
                inverse_distance,
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
        QX = [fft_sliding_dot_product(self.X_[i], X) for i in range(len(self.X_))]
        if self.metadata_["unequal_length"]:
            QX = List(QX)
        else:
            QX = np.asarray(QX)

        if self.normalise:
            distance_profiles = _normalised_squared_distance_profile(
                QX,
                self.X_means_,
                self.X_stds_,
                X.mean(axis=1),
                X.std(axis=1),
                self.length,
            )
        else:
            distance_profiles = _squared_distance_profile(
                QX,
                self.X_,
                X,
            )

        if not self.metadata_["unequal_length"]:
            distance_profiles = np.asarray(distance_profiles)
        return distance_profiles


@njit(cache=True, fastmath=True)
def _stomp_normalised(
    X,
    T,
    XdotT,
    X_means,
    X_stds,
    T_means,
    T_stds,
    L,
    T_index,
    k,
    threshold,
    allow_neighboring_matches,
    exclusion_size,
    inverse_distance,
):
    """
    Compute the Matrix Profile using the STOMP algorithm with normalised distances.

    X:  np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a TypedList
        of 2D arrays of shape (n_channels, n_timepoints)
    T : np.ndarray, 2D array of shape (n_channels, series_length)
        The series used for similarity search. Note that series_length can be equal,
        superior or inferior to n_timepoints, it doesn't matter.
    XdotT : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - L + 1)
        Precomputed dot products between each time series in X and the query series T.
    X_means : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - L + 1)
        Means of each subsequences of X of size L. Should be a numba TypedList if X is
        unequal length.
    X_stds : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - L + 1)
        Stds of each subsequences of X of size L. Should be a numba TypedList if X is
        unequal length.
    T_means : np.ndarray, 2D array of shape (n_channels, n_timepoints - L + 1)
        Means of each subsequences of T of size L.
    T_stds : np.ndarray, 2D array of shape (n_channels, n_timepoints - L + 1)
        Stds of each subsequences of T of size L.
    L : int
        Length of the subsequences used for the distance computation.
    T_index : int,
        If ``T`` is a series of the database given in fit, specify its index
        in ``X_``. If specified, each query of this series won't be able to
        match with its neighboring subsequences.
    k : int,
        The number of best matches to return during predict for each subsequence.
    threshold : float
        The number of best matches to return during predict for each subsequence.
    allow_neighboring_matches : bool
        Wheter the top-k candidates can be neighboring subsequences.
    exclusion_size : int
        The size of the exclusion zone used to prevent returning as top k candidates
        the ones that are close to each other (for example i and i+1).
        It is used to define a region between
        :math:`id_timestamp - exclusion_size` and
        :math:`id_timestamp + exclusion_size` which cannot be returned
        as best match if :math:`id_timestamp` was already selected. By default,
        the value None means that this is not used.
    inverse_distance : bool
        If True, the matching will be made on the inverse of the distance, and thus, the
        worst matches to the query will be returned instead of the best ones.

    Returns
    -------
    tuple of np.ndarray
        - MP : array of shape (series_length - L + 1,)
          Matrix profile distances for each query subsequence.
        - IP : array of shape (series_length - L + 1,)
          Indexes of the top matches for each query subsequence.
    """
    n_queries = T.shape[1] - L + 1
    MP = List()
    IP = List()

    for i_q in range(n_queries):
        dist_profiles = _normalised_squared_distance_profile(
            XdotT, X_means, X_stds, T_means[:, i_q], T_stds[:, i_q], L
        )
        if i_q + 1 < n_queries:
            for i_x in range(len(X)):
                XdotT[i_x] = _update_dot_products_one_series(
                    X[i_x], T, XdotT[i_x], L, i_q + 1
                )

        if inverse_distance:
            dist_profiles = _inverse_distance_profile_list(dist_profiles)

        # Deal with self-matches
        if T_index is not None:
            _max_timestamp = X[T_index].shape[1] - L
            ub = min(i_q + exclusion_size, _max_timestamp)
            lb = max(0, i_q - exclusion_size)
            dist_profiles[T_index][lb:ub] = np.inf

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


@njit(cache=True, fastmath=True)
def _stomp(
    X,
    T,
    XdotT,
    L,
    T_index,
    k,
    threshold,
    allow_neighboring_matches,
    exclusion_size,
    inverse_distance,
):
    """
    Compute the Matrix Profile using the STOMP algorithm with non-normalised distances.

    X:  np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a TypedList
        of 2D arrays of shape (n_channels, n_timepoints)
    T : np.ndarray, 2D array of shape (n_channels, series_length)
        The series used for similarity search. Note that series_length can be equal,
        superior or inferior to n_timepoints, it doesn't matter.
    XdotT : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - L + 1)
        Precomputed dot products between each time series in X and the query series T.
    L : int
        Length of the subsequences used for the distance computation.
    T_index : int,
        If ``T`` is a series of the database given in fit, specify its index
        in ``X_``. If specified, each query of this series won't be able to
        match with its neighboring subsequences.
    k : int,
        The number of best matches to return during predict for each subsequence.
    threshold : float
        The number of best matches to return during predict for each subsequence.
    allow_neighboring_matches : bool
        Wheter the top-k candidates can be neighboring subsequences.
    exclusion_size : int
        The size of the exclusion zone used to prevent returning as top k candidates
        the ones that are close to each other (for example i and i+1).
        It is used to define a region between
        :math:`id_timestamp - exclusion_size` and
        :math:`id_timestamp + exclusion_size` which cannot be returned
        as best match if :math:`id_timestamp` was already selected. By default,
        the value None means that this is not used.
    inverse_distance : bool
        If True, the matching will be made on the inverse of the distance, and thus, the
        worst matches to the query will be returned instead of the best ones.

    Returns
    -------
    tuple of np.ndarray
        - MP : array of shape (series_length - L + 1,)
          Matrix profile distances for each query subsequence.
        - IP : array of shape (series_length - L + 1,)
          Indexes of the top matches for each query subsequence.
    """
    n_queries = T.shape[1] - L + 1
    MP = List()
    IP = List()

    # For each query of size L in T
    for i_q in range(n_queries):
        Q = T[:, i_q : i_q + L]
        dist_profiles = _squared_distance_profile(XdotT, X, Q)
        # For each series in X compute distance profile to the query
        if i_q + 1 < n_queries:
            for i_x in range(len(X)):
                XdotT[i_x] = _update_dot_products_one_series(
                    X[i_x], T, XdotT[i_x], L, i_q + 1
                )

        if inverse_distance:
            dist_profiles = _inverse_distance_profile_list(dist_profiles)

        # Deal with self-matches
        if T_index is not None:
            _max_timestamp = X[T_index].shape[1] - L
            ub = min(i_q + exclusion_size, _max_timestamp)
            lb = max(0, i_q - exclusion_size)
            dist_profiles[T_index][lb:ub] = np.inf

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


@njit(cache=True, fastmath=True)
def _update_dot_products_one_series(
    X,
    T,
    XT_products,
    L,
    i_query,
):
    """
    Update dot products of the i-th query of size L in T from the dot products of i-1.

    Parameters
    ----------
    X: np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        Input time series on which the sliding dot product is computed.
    T: np.ndarray, 2D array of shape (n_channels, series_length)
        The series used for similarity search. Note that series_length can be equal,
        superior or inferior to n_timepoints, it doesn't matter.
    L : int
        The length of the subsequences considered during the search. This parameter
        cannot be larger than n_timepoints and series_length.
    i_query : int
        Query starting index in T.

    Returns
    -------
    XT_products : np.ndarray of shape (n_cases, n_channels, n_timepoints - L + 1)
        Sliding dot product between the i-th subsequence of size L in T and X.

    """
    n_channels = T.shape[0]
    Q = T[:, i_query : i_query + L]
    n_candidates = X.shape[1] - L + 1

    for i_ft in range(n_channels):
        # first element of all 0 to n-1 candidates * first element of previous query
        _a1 = X[i_ft, : n_candidates - 1] * T[i_ft, i_query - 1]
        # last element of all 1 to n candidates * last element of current query
        _a2 = X[i_ft, L : L - 1 + n_candidates] * T[i_ft, i_query + L - 1]

        XT_products[i_ft, 1:] = XT_products[i_ft, :-1] - _a1 + _a2

        # Compute first dot product
        XT_products[i_ft, 0] = np.sum(Q[i_ft] * X[i_ft, :L])
    return XT_products


@njit(cache=True, fastmath=True, parallel=True)
def _squared_distance_profile(QX, X, Q):
    """
    Compute squared distance profiles between query subsequence and time series.

    Parameters
    ----------
    QX : List of np.ndarray
        List of precomputed dot products between queries and time series, with each
        element corresponding to a different time series.
        Shape of each array is (n_channels, n_timepoints - query_length + 1).
    X : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a numba TypedList
        2D array of shape (n_channels, n_timepoints)
    Q : np.ndarray, 2D array of shape (n_channels, query_length)
        The query used for similarity search.
    mask : np.ndarray, 3D array of shape (n_cases, n_timepoints - query_length + 1)
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed.

    Returns
    -------
    distance_profiles : np.ndarray
        3D array of shape (n_cases, n_timepoints - query_length + 1)
        The distance profile between Q and the input time series X.

    """
    distance_profiles = List()
    query_length = Q.shape[1]

    # Init distance profile array with unequal length support
    for i_instance in range(len(X)):
        profile_length = X[i_instance].shape[1] - query_length + 1
        distance_profiles.append(np.full((profile_length), np.inf))

    for _i_instance in prange(len(QX)):
        # prange cast iterator to unit64 with parallel=True
        i_instance = np.int_(_i_instance)

        distance_profiles[i_instance] = _squared_dist_profile_one_series(
            QX[i_instance], X[i_instance], Q
        )
    return distance_profiles


@njit(cache=True, fastmath=True)
def _squared_dist_profile_one_series(QT, T, Q):
    """
    Compute squared distance profile between query subsequence and a single time series.

    This function calculates the squared distance profile for a single time series by
    leveraging the dot product of the query and time series as well as precomputed sums
    of squares to efficiently compute the squared distances.

    Parameters
    ----------
    QT : np.ndarray, 2D array of shape (n_channels, n_timepoints - query_length + 1)
        The dot product between the query and the time series.
    T : np.ndarray, 2D array of shape (n_channels, series_length)
        The series used for similarity search. Note that series_length can be equal,
        superior or inferior to n_timepoints, it doesn't matter.
    Q : np.ndarray
        2D array of shape (n_channels, query_length) representing query subsequence.

    Returns
    -------
    distance_profile : np.ndarray
        2D array of shape (n_channels, n_timepoints - query_length + 1)
        The squared distance profile between the query and the input time series.
    """
    n_channels, profile_length = QT.shape
    query_length = Q.shape[1]
    _QT = -2 * QT
    distance_profile = np.zeros(profile_length)
    for k in prange(n_channels):
        _sum = 0
        _qsum = 0
        for j in prange(query_length):
            _sum += T[k, j] ** 2
            _qsum += Q[k, j] ** 2

        distance_profile += _qsum + _QT[k]
        distance_profile[0] += _sum
        for i in prange(1, profile_length):
            _sum += T[k, i + (query_length - 1)] ** 2 - T[k, i - 1] ** 2
            distance_profile[i] += _sum
    return distance_profile


@njit(cache=True, fastmath=True, parallel=True)
def _normalised_squared_distance_profile(
    QX, X_means, X_stds, Q_means, Q_stds, query_length
):
    """
    Compute the normalised squared distance profiles between query subsequence and input time series.

    Parameters
    ----------
    QX : List of np.ndarray
        List of precomputed dot products between queries and time series, with each element
        corresponding to a different time series.
        Shape of each array is (n_channels, n_timepoints - query_length + 1).
    X_means : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - query_length + 1)  # noqa: E501
        Means of each subsequences of X of size query_length
    X_stds : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - query_length + 1)  # noqa: E501
        Stds of each subsequences of X of size query_length
    Q_means : np.ndarray, 1D array of shape (n_channels)
        Means of the query q
    Q_stds : np.ndarray, 1D array of shape (n_channels)
        Stds of the query q
    query_length : int
        The length of the query subsequence used for the distance profile computation.

    Returns
    -------
    List of np.ndarray
        List of 2D arrays, each of shape (n_channels, n_timepoints - query_length + 1).
        Each array contains the normalised squared distance profile between the query subsequence and the corresponding time series.
        Entries in the array are set to infinity where the mask is False.
    """
    distance_profiles = List()
    Q_is_constant = Q_stds <= AEON_NUMBA_STD_THRESHOLD
    # Init distance profile array with unequal length support
    for i_instance in range(len(QX)):
        profile_length = QX[i_instance].shape[1]
        distance_profiles.append(np.zeros(profile_length))

    for _i_instance in prange(len(QX)):
        # iterator is uint64 with prange and parallel so cast to int to avoid warnings
        i_instance = np.int64(_i_instance)
        distance_profiles[i_instance] = _normalised_squared_dist_profile_one_series(
            QX[i_instance],
            X_means[i_instance],
            X_stds[i_instance],
            Q_means,
            Q_stds,
            query_length,
            Q_is_constant,
        )
    return distance_profiles


@njit(cache=True, fastmath=True)
def _normalised_squared_dist_profile_one_series(
    QT, T_means, T_stds, Q_means, Q_stds, query_length, Q_is_constant
):
    """
    Compute the z-normalised squared Euclidean distance profile for one time series.

    Parameters
    ----------
    QT : np.ndarray, 2D array of shape (n_channels, n_timepoints - query_length + 1)
        The dot product between the query and the time series.
    T_means : np.ndarray, 1D array of length n_channels
        The mean values of the time series for each channel.
    T_stds : np.ndarray, 2D array of shape (n_channels, profile_length)
        The standard deviations of the time series for each channel and position.
    Q_means : np.ndarray, 1D array of shape (n_channels)
        Means of the query q
    Q_stds : np.ndarray, 1D array of shape (n_channels)
        Stds of the query q
    query_length : int
        The length of the query subsequence used for the distance profile computation.
    Q_is_constant : np.ndarray
        1D array of shape (n_channels,) where each element is a Boolean indicating
        whether the query standard deviation for that channel is less than or equal
        to a specified threshold.

    Returns
    -------
    np.ndarray
        2D array of shape (n_channels, n_timepoints - query_length + 1) containing the
        z-normalised squared distance profile between the query subsequence and the time
        series. Entries are computed based on the z-normalised values, with special
        handling for constant values.
    """
    n_channels, profile_length = QT.shape
    distance_profile = np.zeros(profile_length)

    for i in prange(profile_length):
        Sub_is_constant = T_stds[:, i] <= AEON_NUMBA_STD_THRESHOLD
        for k in prange(n_channels):
            # Two Constant case
            if Q_is_constant[k] and Sub_is_constant[k]:
                _val = 0
            # One Constant case
            elif Q_is_constant[k] or Sub_is_constant[k]:
                _val = query_length
            else:
                denom = query_length * Q_stds[k] * T_stds[k, i]

                p = (QT[k, i] - query_length * (Q_means[k] * T_means[k, i])) / denom
                p = min(p, 1.0)

                _val = abs(2 * query_length * (1.0 - p))
            distance_profile[i] += _val

    return distance_profile
