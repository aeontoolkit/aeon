"""
Tests for stomp algorithm.

We do not test equality for returned indexes due to the unstable nature of argsort
and the fact that the "kind=stable" parameter is not yet supported in numba. We instead
test that the returned index match the expected distance value.
"""

__maintainer__ = ["baraline"]

import numpy as np
import pytest
from numba.typed import List
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

from aeon.similarity_search.subsequence_search._brute_force import (
    _compute_dist_profile,
    _naive_squared_distance_profile,
    _naive_squared_matrix_profile,
)
from aeon.similarity_search.subsequence_search._commons import (
    _extract_top_k_from_dist_profile,
    _inverse_distance_profile_list,
)
from aeon.testing.data_generation import (
    make_example_2d_numpy_series,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.utils.numba.general import (
    get_all_subsequences,
    sliding_mean_std_one_series,
    z_normalise_series_2d,
)

K_VALUES = [1, 3, 5]
NN_MATCHES = [True, False]
INVERSE = [True, False]
NORMALISE = [True, False]


def _get_mean_sdts_inputs(X, Q, L):
    X_means = []
    X_stds = []

    for i_x in range(len(X)):
        _mean, _std = sliding_mean_std_one_series(X[i_x], L, 1)
        X_stds.append(_std)
        X_means.append(_mean)

    Q_means = Q.mean(axis=1)
    Q_stds = Q.std(axis=1)

    return X_means, X_stds, Q_means, Q_stds


def test__compute_dist_profile():
    """Test Euclidean distance with brute force."""
    L = 3
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=L)
    dist_profile = _compute_dist_profile(get_all_subsequences(X, L, 1), Q)
    for i_t in range(X.shape[1] - L + 1):
        assert_almost_equal(dist_profile[i_t], np.sum((X[:, i_t : i_t + L] - Q) ** 2))


@pytest.mark.parametrize("normalise", NORMALISE)
def test__naive_squared_distance_profile(normalise):
    """Test Euclidean distance profile calculation with brute force."""
    L = 3
    X = make_example_3d_numpy(n_cases=3, n_channels=1, n_timepoints=10, return_y=False)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=L)
    dist_profiles = _naive_squared_distance_profile(X, Q, normalise=normalise)

    if normalise:
        Q = z_normalise_series_2d(Q)
    for i_x in range(len(X)):
        for i_t in range(X[i_x].shape[1] - L + 1):
            _x = X[i_x, :, i_t : i_t + L]
            if normalise:
                _x = z_normalise_series_2d(_x)
            assert_almost_equal(dist_profiles[i_x][i_t], np.sum((_x - Q) ** 2))

    # test unequal length and multivariate
    X = List(
        make_example_3d_numpy_list(
            n_cases=3,
            n_channels=2,
            min_n_timepoints=10,
            max_n_timepoints=20,
            return_y=False,
        )
    )

    Q = make_example_2d_numpy_series(n_channels=2, n_timepoints=L)
    dist_profiles = _naive_squared_distance_profile(X, Q, normalise=normalise)
    if normalise:
        Q = z_normalise_series_2d(Q)
    for i_x in range(len(X)):
        for i_t in range(X[i_x].shape[1] - L + 1):
            _x = X[i_x][:, i_t : i_t + L]
            if normalise:
                _x = z_normalise_series_2d(_x)
            assert_almost_equal(dist_profiles[i_x][i_t], np.sum((_x - Q) ** 2))


@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("allow_neighboring_matches", NN_MATCHES)
@pytest.mark.parametrize("inverse_distance", INVERSE)
@pytest.mark.parametrize("normalise", NORMALISE)
def test__naive_squared_matrix_profile(
    k, allow_neighboring_matches, inverse_distance, normalise
):
    """Test brute force matrix profile method."""
    L = 3
    X = List(
        make_example_3d_numpy_list(
            n_cases=3,
            n_channels=2,
            min_n_timepoints=6,
            max_n_timepoints=8,
            return_y=False,
        )
    )
    X_copy = X.copy()
    T = make_example_2d_numpy_series(n_channels=2, n_timepoints=5)
    T_copy = T.copy()
    T_index = None
    threshold = np.inf
    exclusion_size = L
    # MP : distances to best matches for each query
    # IP : Indexes of best matches for each query
    MP, IP = _naive_squared_matrix_profile(
        X,
        T,
        L,
        T_index,
        k,
        threshold,
        allow_neighboring_matches,
        exclusion_size,
        inverse_distance,
        normalise=normalise,
    )
    assert_array_equal(T, T_copy)
    for i in range(len(X)):
        assert_array_equal(X[i], X_copy[i])
    # For each query of size L in T
    for i in range(T.shape[1] - L + 1):
        dist_profiles = _naive_squared_distance_profile(
            X, T[:, i : i + L], normalise=normalise
        )
        # Check that the top matches extracted have the same value that the
        # top matches in the distance profile
        if inverse_distance:
            dist_profiles = _inverse_distance_profile_list(dist_profiles)

        top_k_indexes, top_k_distances = _extract_top_k_from_dist_profile(
            dist_profiles, k, threshold, allow_neighboring_matches, exclusion_size
        )
        # Check that the top matches extracted have the same value that the
        # top matches in the distance profile
        assert_array_almost_equal(MP[i], top_k_distances)

        # Check that the index in IP correspond to a distance profile point
        # with value equal to the corresponding MP point.
        for j, index in enumerate(top_k_indexes):
            assert_almost_equal(MP[i][j], dist_profiles[index[0]][index[1]])
