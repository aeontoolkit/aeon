"""Test _commons.py functions."""

__maintainer__ = ["baraline"]
import numpy as np
import pytest
from numba.typed import List
from numpy.testing import assert_, assert_array_almost_equal

from aeon.similarity_search.series._commons import (
    _extract_top_k_from_dist_profile,
    _extract_top_k_motifs,
    _extract_top_r_motifs,
    _inverse_distance_profile,
    _update_dot_products,
    fft_sliding_dot_product,
    get_ith_products,
)
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
)

K_VALUES = [1, 3, 5]
THRESHOLDS = [np.inf, 1.5]
NN_MATCHES = [False, True]
EXCLUSION_SIZE = [3, 5]


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


def test__update_dot_products():
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
        current_product = _update_dot_products(
            X,
            T,
            current_product,
            L,
            i_query,
        )
        assert_array_almost_equal(new_product, current_product)


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


def test__inverse_distance_profile():
    """Test method to inverse a TypedList of distance profiles."""
    X = make_example_1d_numpy()
    X_inv = _inverse_distance_profile(X)
    assert_array_almost_equal(1 / (X + 1e-8), X_inv)


def test__extract_top_k_motifs():
    """Test motif extraction based on max distance."""
    MP = List(
        [
            [1.0, 2.0],
            [1.0, 4.0],
            [0.5, 0.9],
            [0.6, 0.7],
        ]
    )
    IP = List(
        [
            [1, 2],
            [1, 4],
            [0, 3],
            [0, 7],
        ]
    )
    MP_k, IP_k = _extract_top_k_motifs(MP, IP, 2, True, 0)
    assert_(len(MP_k) == 2)
    assert_(MP_k[0] == [0.6, 0.7])
    assert_(IP_k[0] == [0, 7])
    assert_(MP_k[1] == [0.5, 0.9])
    assert_(IP_k[1] == [0, 3])


def test__extract_top_r_motifs():
    """Test motif extraction based on motif set cardinality."""
    MP = List(
        [
            [1.0, 1.5, 2.0, 1.5],
            [1.0, 4.0],
            [0.5, 0.9, 1.0],
            [0.6, 0.7],
        ]
    )
    IP = List(
        [
            [1, 2, 3, 4],
            [1, 4],
            [0, 3, 6],
            [0, 7],
        ]
    )
    MP_k, IP_k = _extract_top_r_motifs(MP, IP, 2, True, 0)
    assert_(len(MP_k) == 2)
    assert_(MP_k[0] == [1.0, 1.5, 2.0, 1.5])
    assert_(IP_k[0] == [1, 2, 3, 4])
    assert_(MP_k[1] == [0.5, 0.9, 1.0])
    assert_(IP_k[1] == [0, 3, 6])


@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("allow_nn_matches", NN_MATCHES)
@pytest.mark.parametrize("exclusion_size", EXCLUSION_SIZE)
def test__extract_top_k_from_dist_profile(
    k, threshold, allow_nn_matches, exclusion_size
):
    """Test method to esxtract the top k candidates from a list of distance profiles."""
    X = make_example_1d_numpy(n_timepoints=30)
    X_sort = np.argsort(X)
    exclusion_size = 3
    top_k_indexes, top_k_distances = _extract_top_k_from_dist_profile(
        X, k, threshold, allow_nn_matches, exclusion_size
    )

    if len(top_k_indexes) == 0 or len(top_k_distances) == 0:
        raise AssertionError("_extract_top_k_from_dist_profile returned empty list")
    for i, index in enumerate(top_k_indexes):
        assert_(X[index] == top_k_distances[i])

    assert_(np.all(top_k_distances <= threshold))

    if allow_nn_matches:
        assert_(np.all(top_k_distances <= X_sort[len(top_k_indexes) - 1]))

    if not allow_nn_matches:
        same_X = np.sort(top_k_indexes)
        if len(same_X) > 1:
            assert_(np.all(np.diff(same_X) >= exclusion_size))
