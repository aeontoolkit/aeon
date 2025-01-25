"""
Tests for stomp algorithm.

We do not test equality for returned indexes due to the unstable nature of argsort
and the fact that the "kind=stable" parameter is not yet supported in numba. We instead
test that the returned index match the expected distance value.
"""

__maintainer__ = ["baraline"]

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from aeon.similarity_search.series._commons import (
    _extract_top_k_from_dist_profile,
    _inverse_distance_profile,
    get_ith_products,
)
from aeon.similarity_search.series.motifs._stomp import _stomp, _stomp_normalized
from aeon.similarity_search.series.neighbors._dummy import (
    _naive_squared_distance_profile,
)
from aeon.testing.data_generation import make_example_2d_numpy_series
from aeon.utils.numba.general import (
    get_all_subsequences,
    sliding_mean_std_one_series,
    z_normalise_series_3d,
)

MOTIFS_SIZE_VALUES = [1, 3]
THRESHOLD = [np.inf, 0.75]
THRESHOLD_NORM = [np.inf, 4.5]
NN_MATCHES = [True, False]
INVERSE = [True, False]


@pytest.mark.parametrize("motif_size", MOTIFS_SIZE_VALUES)
@pytest.mark.parametrize("threshold", THRESHOLD)
@pytest.mark.parametrize("allow_trivial_matches", NN_MATCHES)
@pytest.mark.parametrize("inverse_distance", INVERSE)
def test__stomp(motif_size, threshold, allow_trivial_matches, inverse_distance):
    """Test STOMP method."""
    L = 3

    X_A = make_example_2d_numpy_series(
        n_channels=2,
        n_timepoints=10,
    )
    X_B = make_example_2d_numpy_series(n_channels=2, n_timepoints=10)
    AdotB = get_ith_products(X_B, X_A, L, 0)

    exclusion_size = L
    # MP : distances to best matches for each query
    # IP : Indexes of best matches for each query
    MP, IP = _stomp(
        X_A,
        X_B,
        AdotB,
        L,
        motif_size,
        threshold,
        allow_trivial_matches,
        exclusion_size,
        inverse_distance,
        False,
    )
    # For each query of size L in T
    X_B_subs = get_all_subsequences(X_B, L, 1)
    X_A_subs = get_all_subsequences(X_A, L, 1)
    for i in range(X_A.shape[1] - L + 1):
        dist_profile = _naive_squared_distance_profile(X_B_subs, X_A_subs[i])
        # Check that the top matches extracted have the same value that the
        # top matches in the distance profile
        if inverse_distance:
            dist_profile = _inverse_distance_profile(dist_profile)

        top_k_indexes, top_k_distances = _extract_top_k_from_dist_profile(
            dist_profile, motif_size, threshold, allow_trivial_matches, exclusion_size
        )
        # Check that the top matches extracted have the same value that the
        # top matches in the distance profile
        assert_array_almost_equal(MP[i], top_k_distances)

        # Check that the index in IP correspond to a distance profile point
        # with value equal to the corresponding MP point.
        for j, index in enumerate(top_k_indexes):
            assert_almost_equal(MP[i][j], dist_profile[index])


@pytest.mark.parametrize("motif_size", MOTIFS_SIZE_VALUES)
@pytest.mark.parametrize("threshold", THRESHOLD_NORM)
@pytest.mark.parametrize("allow_trivial_matches", NN_MATCHES)
@pytest.mark.parametrize("inverse_distance", INVERSE)
def test__stomp_normalised(
    motif_size, threshold, allow_trivial_matches, inverse_distance
):
    """Test STOMP normalised method."""
    L = 3

    X_A = make_example_2d_numpy_series(
        n_channels=2,
        n_timepoints=10,
    )
    X_B = make_example_2d_numpy_series(n_channels=2, n_timepoints=10)
    X_A_means, X_A_stds = sliding_mean_std_one_series(X_A, L, 1)
    X_B_means, X_B_stds = sliding_mean_std_one_series(X_B, L, 1)
    AdotB = get_ith_products(X_B, X_A, L, 0)

    exclusion_size = L
    # MP : distances to best matches for each query
    # IP : Indexes of best matches for each query
    MP, IP = _stomp_normalized(
        X_A,
        X_B,
        AdotB,
        X_A_means,
        X_A_stds,
        X_B_means,
        X_B_stds,
        L,
        motif_size,
        threshold,
        allow_trivial_matches,
        exclusion_size,
        inverse_distance,
        False,
    )
    # For each query of size L in T
    X_B_subs = z_normalise_series_3d(get_all_subsequences(X_B, L, 1))
    X_A_subs = z_normalise_series_3d(get_all_subsequences(X_A, L, 1))
    for i in range(X_A.shape[1] - L + 1):
        dist_profile = _naive_squared_distance_profile(X_B_subs, X_A_subs[i])
        # Check that the top matches extracted have the same value that the
        # top matches in the distance profile
        if inverse_distance:
            dist_profile = _inverse_distance_profile(dist_profile)
        top_k_indexes, top_k_distances = _extract_top_k_from_dist_profile(
            dist_profile, motif_size, threshold, allow_trivial_matches, exclusion_size
        )

        # Check that the top matches extracted have the same value that the
        # top matches in the distance profile
        assert_array_almost_equal(MP[i], top_k_distances)

        # Check that the index in IP correspond to a distance profile point
        # with value equal to the corresponding MP point.
        for j, index in enumerate(top_k_indexes):
            assert_almost_equal(MP[i][j], dist_profile[index])
