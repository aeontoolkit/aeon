"""Helper and common function for similarity search estimators and functions."""

__maintainer__ = ["baraline"]

import numpy as np
from numba import njit, prange
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


@njit(cache=True, fastmath=True, parallel=True)
def _inverse_distance_profile_list(dist_profiles):
    for i in prange(len(dist_profiles)):
        dist_profiles[i] = 1 / (dist_profiles[i] + 1e-8)
    return dist_profiles


@njit(cache=True)
def _extract_top_k_from_dist_profile(
    dist_profiles,
    k,
    threshold,
    allow_neighboring_matches,
    exclusion_size,
):
    """
    Given an array (or list) of distance profiles, extract the top k lower distances.

    Parameters
    ----------
    dist_profiles : np.ndarray, shape = (n_samples, n_timepoints - length + 1)
        A collection of distance profiles computed from ``n_samples`` time series of
        size ``n_timepoints``, giving distance profiles of length
        ``n_timepoints - length + 1``, with ``length`` the size of the query used to
        compute the distance profiles.
    k : int
        Number of best matches to return
    threshold : float
        A threshold on the distances of the best matches. To be returned, a candidate
        must have a distance bellow this threshold. This can reduce the number of
        returned matches to be bellow ``k``
    allow_neighboring_matches : bool
        Wheter to allow returning matches that are in the same neighborhood.
    exclusion_size : int
        The size of the exlusion size to apply when ``allow_neighboring_matches`` is
        False. It is applied on both side of existing matches (+/- their indexes).

    Returns
    -------
    top_k_indexes : np.ndarray, shape = (k, 2)
        The indexes of the best matches in ``distance_profiles``.
    top_k_distances : np.ndarray, shape = (k)
        The distances of the best matches.

    """
    top_k_indexes = np.zeros((2 * k, 2), dtype=np.int64) - 1
    top_k_distances = np.full(2 * k, np.inf)
    for i_profile in range(len(dist_profiles)):
        # Extract top-k without neighboring matches
        if not allow_neighboring_matches:
            _sorted_indexes = np.argsort(dist_profiles[i_profile])
            _top_k_indexes = np.zeros(k, dtype=np.int64) - 1
            _current_k = 0
            _current_j = 0
            # Until we extract k value or explore all the array
            while _current_k < k and _current_j < len(_sorted_indexes):
                _insert = True
                # Check for validity with each previously inserted
                for i_k in range(_current_k):
                    ub = min(
                        _top_k_indexes[i_k] + exclusion_size,
                        len(dist_profiles[i_profile]),
                    )
                    lb = max(_top_k_indexes[i_k] - exclusion_size, 0)
                    if (
                        _sorted_indexes[_current_j] >= lb
                        and _sorted_indexes[_current_j] <= ub
                    ):
                        _insert = False
                        break

                if _insert:
                    _top_k_indexes[_current_k] = _sorted_indexes[_current_j]
                    _current_k += 1
                _current_j += 1

            _top_k_indexes = _top_k_indexes[:_current_k]
            _top_k_distances = dist_profiles[i_profile][_top_k_indexes]
        # Extract top-k with neighboring matches
        else:
            _top_k_indexes = np.argsort(dist_profiles[i_profile])[:k]
            _top_k_distances = dist_profiles[i_profile][_top_k_indexes]

        # Select overall top k by using the buffer array of size 2*k
        # Inset top from current sample
        top_k_distances[k : k + len(_top_k_distances)] = _top_k_distances
        top_k_indexes[k : k + len(_top_k_distances), 1] = _top_k_indexes
        top_k_indexes[k : k + len(_top_k_distances), 0] = i_profile

        # Sort overall
        idx = np.argsort(top_k_distances)
        # Keep top k overall
        top_k_distances[:k] = top_k_distances[idx[:k]]
        top_k_indexes[:k] = top_k_indexes[idx[:k]]

        top_k_distances[k:] = np.inf

    # get the actual number of extracted values and apply threshold
    true_k = 0
    for i in range(k):
        # if top_k is inf, it means that no value was extracted
        if top_k_distances[i] != np.inf and top_k_distances[i] <= threshold:
            true_k += 1
        else:
            break

    return top_k_indexes[:true_k], top_k_distances[:true_k]
