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
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from aeon.similarity_search.subsequence_search._commons import (
    _extract_top_k_from_dist_profile,
    _inverse_distance_profile_list,
    get_ith_products,
)
from aeon.similarity_search.subsequence_search._stomp import (
    _normalised_squared_dist_profile_one_series,
    _normalised_squared_distance_profile,
    _squared_dist_profile_one_series,
    _squared_distance_profile,
    _stomp,
    _stomp_normalised,
    _update_dot_products_one_series,
)
from aeon.testing.data_generation import (
    make_example_2d_numpy_series,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.utils.numba.general import (
    sliding_mean_std_one_series,
    z_normalise_series_2d_with_mean_std,
)

K_VALUES = [1, 3, 5]
NN_MATCHES = [True, False]
INVERSE = [True, False]


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


def test__update_dot_products_one_series():
    """Test the _update_dot_product function."""
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=20)
    T = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    L = 7
    current_product = get_ith_products(X, T, L, 0)
    for i_query in range(1, T.shape[1] - L + 1):
        new_product = get_ith_products(
            X,
            T,
            L,
            i_query,
        )
        current_product = _update_dot_products_one_series(
            X,
            T,
            current_product,
            L,
            i_query,
        )
        assert_array_almost_equal(new_product, current_product)


def test__squared_dist_profile_one_series():
    """Test Euclidean distance."""
    L = 3
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=L)
    QX = get_ith_products(X, Q, L, 0)
    dist_profile = _squared_dist_profile_one_series(QX, X, Q)
    for i_t in range(X.shape[1] - L + 1):
        assert_almost_equal(dist_profile[i_t], np.sum((X[:, i_t : i_t + L] - Q) ** 2))


def test__normalised_squared_dist_profile_one_series():
    """Test Euclidean distance."""
    L = 3
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=L)
    QX = get_ith_products(X, Q, L, 0)
    X_mean, X_std = sliding_mean_std_one_series(X, L, 1)
    Q_mean = Q.mean(axis=1)
    Q_std = Q.std(axis=1)

    dist_profile = _normalised_squared_dist_profile_one_series(
        QX, X_mean, X_std, Q_mean, Q_std, L, Q.std(axis=1) <= 0
    )
    Q = z_normalise_series_2d_with_mean_std(Q, Q_mean, Q_std)
    for i_t in range(X.shape[1] - L + 1):
        S = z_normalise_series_2d_with_mean_std(
            X[:, i_t : i_t + L], X_mean[:, i_t], X_std[:, i_t]
        )
        assert_almost_equal(dist_profile[i_t], np.sum((S - Q) ** 2))


def test__squared_distance_profile():
    """Test Euclidean distance profile calculation."""
    L = 3
    X = make_example_3d_numpy(n_cases=3, n_channels=1, n_timepoints=10, return_y=False)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=L)
    QX = np.asarray([get_ith_products(X[i_x], Q, L, 0) for i_x in range(len(X))])
    dist_profiles = _squared_distance_profile(QX, X, Q)
    for i_x in range(len(X)):
        for i_t in range(X[i_x].shape[1] - L + 1):
            assert_almost_equal(
                dist_profiles[i_x][i_t], np.sum((X[i_x, :, i_t : i_t + L] - Q) ** 2)
            )

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
    QX = List([get_ith_products(X[i_x], Q, L, 0) for i_x in range(len(X))])
    dist_profiles = _squared_distance_profile(QX, X, Q)
    for i_x in range(len(X)):
        for i_t in range(X[i_x].shape[1] - L + 1):
            assert_almost_equal(
                dist_profiles[i_x][i_t], np.sum((X[i_x][:, i_t : i_t + L] - Q) ** 2)
            )


def test__normalised_squared_distance_profile():
    """Test Euclidean distance profile calculation."""
    L = 3
    X = make_example_3d_numpy(n_cases=3, n_channels=1, n_timepoints=10, return_y=False)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=L)
    QX = np.asarray([get_ith_products(X[i_x], Q, L, 0) for i_x in range(len(X))])

    X_means, X_stds, Q_means, Q_stds = _get_mean_sdts_inputs(X, Q, L)

    X_means = np.asarray(X_means)
    X_stds = np.asarray(X_stds)

    dist_profiles = _normalised_squared_distance_profile(
        QX, X_means, X_stds, Q_means, Q_stds, L
    )

    Q_norm = z_normalise_series_2d_with_mean_std(Q, Q_means, Q_stds)
    for i_x in range(len(X)):
        for i_t in range(X[i_x].shape[1] - L + 1):
            X_sub_norm = z_normalise_series_2d_with_mean_std(
                X[i_x, :, i_t : i_t + L], X_means[i_x][:, i_t], X_stds[i_x][:, i_t]
            )
            assert_almost_equal(
                dist_profiles[i_x][i_t], np.sum((X_sub_norm - Q_norm) ** 2)
            )

    # test unequal length and multivariate
    X = List(
        make_example_3d_numpy_list(
            n_cases=5,
            n_channels=2,
            min_n_timepoints=10,
            max_n_timepoints=20,
            return_y=False,
        )
    )
    Q = make_example_2d_numpy_series(n_channels=2, n_timepoints=L)

    QX = List([get_ith_products(X[i_x], Q, L, 0) for i_x in range(len(X))])

    X_means, X_stds, Q_means, Q_stds = _get_mean_sdts_inputs(X, Q, L)
    # Convert to numba typed list
    X_means = List(X_means)
    X_stds = List(X_stds)

    dist_profiles = _normalised_squared_distance_profile(
        QX, X_means, X_stds, Q_means, Q_stds, L
    )

    Q_norm = z_normalise_series_2d_with_mean_std(Q, Q_means, Q_stds)
    for i_x in range(len(X)):
        for i_t in range(X[i_x].shape[1] - L + 1):
            X_sub_norm = z_normalise_series_2d_with_mean_std(
                X[i_x][:, i_t : i_t + L], X_means[i_x][:, i_t], X_stds[i_x][:, i_t]
            )
            assert_almost_equal(
                dist_profiles[i_x][i_t], np.sum((X_sub_norm - Q_norm) ** 2)
            )


@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("allow_neighboring_matches", NN_MATCHES)
@pytest.mark.parametrize("inverse_distance", INVERSE)
def test__stomp(k, allow_neighboring_matches, inverse_distance):
    """Test STOMP method."""
    L = 3

    X = make_example_3d_numpy_list(
        n_cases=3,
        n_channels=2,
        min_n_timepoints=6,
        max_n_timepoints=8,
        return_y=False,
    )
    T = make_example_2d_numpy_series(n_channels=2, n_timepoints=5)
    XdotT = List([get_ith_products(X[i_x], T, L, 0) for i_x in range(len(X))])

    T_index = None
    threshold = np.inf
    exclusion_size = L
    # MP : distances to best matches for each query
    # IP : Indexes of best matches for each query
    MP, IP = _stomp(
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
    )
    # For each query of size L in T
    for i in range(T.shape[1] - L + 1):
        dist_profiles = _squared_distance_profile(
            List([get_ith_products(X[i_x], T, L, i) for i_x in range(len(X))]),
            X,
            T[:, i : i + L],
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


@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("allow_neighboring_matches", NN_MATCHES)
@pytest.mark.parametrize("inverse_distance", INVERSE)
def test__stomp_normalised(k, allow_neighboring_matches, inverse_distance):
    """Test STOMP normalised method."""
    L = 3
    X = make_example_3d_numpy_list(
        n_cases=3,
        n_channels=2,
        min_n_timepoints=6,
        max_n_timepoints=8,
        return_y=False,
    )
    T = make_example_2d_numpy_series(n_channels=2, n_timepoints=5)

    XdotT = List([get_ith_products(X[i_x], T, L, 0) for i_x in range(len(X))])

    T_index = None
    threshold = np.inf
    exclusion_size = L
    X_means, X_stds, _, _ = _get_mean_sdts_inputs(X, T, L)
    T_means, T_stds = sliding_mean_std_one_series(T, L, 1)
    # MP : distances to best matches for each query
    # IP : Indexes of best matches for each query
    MP, IP = _stomp_normalised(
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
    )
    # For each query of size L in T
    for i in range(T.shape[1] - L + 1):
        dist_profiles = _normalised_squared_distance_profile(
            List([get_ith_products(X[i_x], T, L, i) for i_x in range(len(X))]),
            X_means,
            X_stds,
            T_means[:, i],
            T_stds[:, i],
            L,
        )

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
