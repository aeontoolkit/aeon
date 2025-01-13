"""Test _commons.py functions."""

__maintainer__ = ["baraline"]
import numpy as np
import pytest
from numba.typed import List
from numpy.testing import assert_, assert_array_almost_equal

from aeon.similarity_search.subsequence_search._commons import (
    _extract_top_k_from_dist_profile,
    _inverse_distance_profile_list,
    fft_sliding_dot_product,
    get_ith_products,
)
from aeon.testing.data_generation import (
    make_example_2d_numpy_list,
    make_example_2d_numpy_series,
)

K_VALUES = [1, 3, 5]
THRESHOLDS = [np.inf, 0.7]
NN_MATCHES = [False, True]


def test_fft_sliding_dot_product():
    """Test the fft_sliding_dot_product function."""
    L = 4
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=L)

    values = fft_sliding_dot_product(X, Q)
    # Compare values[0] only as input is univariate
    assert_array_almost_equal(
        values[0],
        [np.dot(Q[0], X[0, i : i + L]) for i in range(X.shape[1] - L + 1)],
    )


def test_get_ith_products():
    """Test i-th dot product of a subsequence of size L."""
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    L = 5

    values = get_ith_products(X, Q, L, 0)
    # Compare values[0] only as input is univariate
    assert_array_almost_equal(
        values[0],
        [np.dot(Q[0, 0:L], X[0, i : i + L]) for i in range(X.shape[1] - L + 1)],
    )

    values = get_ith_products(X, Q, L, 4)
    # Compare values[0] only as input is univariate
    assert_array_almost_equal(
        values[0],
        [np.dot(Q[0, 4 : 4 + L], X[0, i : i + L]) for i in range(X.shape[1] - L + 1)],
    )


def test__inverse_distance_profile_list():
    """Test method to inverse a TypedList of distance profiles."""
    X = make_example_2d_numpy_list(n_cases=2, return_y=False)
    T = _inverse_distance_profile_list(List(X))
    assert_array_almost_equal(1 / (X[0] + 1e-8), T[0])
    assert_array_almost_equal(1 / (X[1] + 1e-8), T[1])


@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("allow_nn_matches", NN_MATCHES)
def test__extract_top_k_from_dist_profile(k, threshold, allow_nn_matches):
    """Test method to esxtract the top k candidates from a list of distance profiles."""
    X = make_example_2d_numpy_list(
        n_cases=2, min_n_timepoints=5, max_n_timepoints=7, return_y=False
    )
    X_sort = [X[i][np.argsort(X[i])] for i in range(len(X))]

    top_k_indexes, top_k_distances = _extract_top_k_from_dist_profile(
        X, k, threshold, allow_nn_matches, 3
    )
    for i, index in enumerate(top_k_indexes):
        assert_(X[index[0]][index[1]] == top_k_distances[i])
    assert_(np.all(top_k_distances <= threshold))
    if allow_nn_matches:
        for i in range(len(X)):
            assert_(np.all(top_k_distances <= X_sort[i][k - 1]))
    if not allow_nn_matches:
        for i_x in range(len(X)):
            # test same index X respect exclusion
            same_X = [
                top_k_indexes[i][1]
                for i in range(len(top_k_indexes))
                if top_k_indexes[i][0] == i_x
            ]
            same_X = np.sort(same_X)
            if len(same_X) > 1:
                assert_(np.all(np.diff(same_X) >= 3))
