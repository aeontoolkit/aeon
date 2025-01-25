"""Helper and common function for similarity search series estimators."""

__maintainer__ = ["baraline"]

import numpy as np
from numba import njit
from scipy.signal import convolve


def fft_sliding_dot_product(X, q):
    """
    Use FFT convolution to calculate the sliding window dot product.

    This function applies the Fast Fourier Transform (FFT) to efficiently compute
    the sliding dot product between the input time series `X` and the query `q`.
    The dot product is computed for each channel individually. The sliding window
    approach ensures that the dot product is calculated for every possible subsequence
    of `X` that matches the length of `q`

    Parameters
    ----------
    X : array, shape=(n_channels, n_timepoints)
        Input time series
    q : array, shape=(n_channels, query_length)
        Input query

    Returns
    -------
    out : np.ndarray, 2D array of shape (n_channels, n_timepoints - query_length + 1)
        Sliding dot product between q and X.
    """
    n_channels, n_timepoints = X.shape
    query_length = q.shape[1]
    out = np.zeros((n_channels, n_timepoints - query_length + 1))
    for i in range(n_channels):
        out[i, :] = convolve(np.flipud(q[i, :]), X[i, :], mode="valid").real
    return out


def get_ith_products(X, T, L, ith):
    """
    Compute dot products between X and the i-th subsequence of size L in T.

    Parameters
    ----------
    X : array, shape = (n_channels, n_timepoints_X)
        Input data.
    T : array, shape =  (n_channels, n_timepoints_T)
        Data containing the query.
    L : int
        Overall query length.
    ith : int
        Query starting index in T.

    Returns
    -------
    np.ndarray, 2D array of shape (n_channels, n_timepoints_X - L + 1)
        Sliding dot product between the i-th subsequence of size L in T and X.

    """
    return fft_sliding_dot_product(X, T[:, ith : ith + L])


@njit(cache=True, fastmath=True)
def _inverse_distance_profile(dist_profile):
    return 1 / (dist_profile + 1e-8)


@njit(cache=True)
def _extract_top_k_from_dist_profile(
    dist_profile,
    k,
    threshold,
    allow_trivial_matches,
    exclusion_size,
):
    """
    Given a distance profiles, extract the top k lower distances.

    Parameters
    ----------
    dist_profile : np.ndarray, shape = (n_timepoints - length + 1)
        A distance profile of length ``n_timepoints - length + 1``, with
        ``length`` the size of the query used to compute the distance profiles.
    k : int
        Number of best matches to return
    threshold : float
        A threshold on the distances of the best matches. To be returned, a candidate
        must have a distance bellow this threshold. This can reduce the number of
        returned matches to be bellow ``k``
    allow_trivial_matches : bool
        Wheter to allow returning matches that are in the same neighborhood.
    exclusion_size : int
        The size of the exlusion size to apply when ``allow_trivial_matches`` is
        False. It is applied on both side of existing matches (+/- their indexes).

    Returns
    -------
    top_k_indexes : np.ndarray, shape = (k)
        The indexes of the best matches in ``distance_profile``.
    top_k_distances : np.ndarray, shape = (k)
        The distances of the best matches.

    """
    top_k_indexes = np.zeros(k, dtype=np.int64) - 1
    top_k_distances = np.full(k, np.inf, dtype=np.float64)
    ub = np.full(k, np.inf)
    lb = np.full(k, -1.0)
    # Could be optimized by using argpartition
    sorted_indexes = np.argsort(dist_profile)
    _current_k = 0
    if not allow_trivial_matches:
        _current_j = 0
        # Until we extract k value or explore all the array or until dist is > threshold
        while _current_k < k and _current_j < len(sorted_indexes):
            # if we didn't insert anything or there is a conflict in lb/ub
            if _current_k > 0 and np.any(
                (sorted_indexes[_current_j] >= lb[:_current_k])
                & (sorted_indexes[_current_j] <= ub[:_current_k])
            ):
                pass
            else:
                _idx = sorted_indexes[_current_j]
                if dist_profile[_idx] <= threshold:
                    top_k_indexes[_current_k] = _idx
                    top_k_distances[_current_k] = dist_profile[_idx]
                    ub[_current_k] = min(
                        top_k_indexes[_current_k] + exclusion_size,
                        len(dist_profile),
                    )
                    lb[_current_k] = max(top_k_indexes[_current_k] - exclusion_size, 0)
                    _current_k += 1
                else:
                    break
            _current_j += 1
    else:
        _current_k += min(k, len(dist_profile))
        dist_profile = dist_profile[sorted_indexes[:_current_k]]
        dist_profile = dist_profile[dist_profile <= threshold]
        _current_k = len(dist_profile)

        top_k_indexes[:_current_k] = sorted_indexes[:_current_k]
        top_k_distances[:_current_k] = dist_profile[:_current_k]

    return top_k_indexes[:_current_k], top_k_distances[:_current_k]


# Could add aggregation function as parameter instead of just max
def _extract_top_k_motifs(MP, IP, k, allow_trivial_matches, exclusion_size):
    criterion = np.zeros(len(MP))
    for i in range(len(MP)):
        if len(MP[i]) > 0:
            criterion[i] = max(MP[i])
        else:
            criterion[i] = np.inf

    idx, _ = _extract_top_k_from_dist_profile(
        criterion, k, np.inf, allow_trivial_matches, exclusion_size
    )
    return [MP[i] for i in idx], [IP[i] for i in idx]


def _extract_top_r_motifs(MP, IP, k, allow_trivial_matches, exclusion_size):
    criterion = np.zeros(len(MP))
    for i in range(len(MP)):
        criterion[i] = len(MP[i])
    idx, _ = _extract_top_k_from_dist_profile(
        _inverse_distance_profile(criterion),
        k,
        np.inf,
        allow_trivial_matches,
        exclusion_size,
    )
    return [MP[i] for i in idx], [IP[i] for i in idx]


@njit(cache=True, fastmath=True)
def _update_dot_products(
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
    X: np.ndarray, 2D array of shape (n_channels, n_timepoints)
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
    XT_products : np.ndarray of shape (n_channels, n_timepoints - L + 1)
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
