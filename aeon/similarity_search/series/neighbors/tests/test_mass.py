"""Tests for MASS algorithm."""

__maintainer__ = ["baraline"]

import numpy as np
from numpy.testing import assert_almost_equal

from aeon.similarity_search.series._commons import fft_sliding_dot_product
from aeon.similarity_search.series.neighbors._mass import (
    _normalized_squared_distance_profile,
    _squared_distance_profile,
)
from aeon.testing.data_generation import make_example_2d_numpy_series
from aeon.utils.numba.general import sliding_mean_std_one_series, z_normalise_series_2d


def test__squared_distance_profile():
    """Test squared distance profile."""
    L = 3
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=L)
    QX = fft_sliding_dot_product(X, Q)
    dist_profile = _squared_distance_profile(QX, X, Q)
    for i_t in range(X.shape[1] - L + 1):
        assert_almost_equal(dist_profile[i_t], np.sum((X[:, i_t : i_t + L] - Q) ** 2))


def test__normalized_squared_distance_profile():
    """Test Euclidean distance."""
    L = 3
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=L)
    QX = fft_sliding_dot_product(X, Q)
    X_mean, X_std = sliding_mean_std_one_series(X, L, 1)
    Q_mean = Q.mean(axis=1)
    Q_std = Q.std(axis=1)

    dist_profile = _normalized_squared_distance_profile(
        QX, X_mean, X_std, Q_mean, Q_std, L
    )
    Q = z_normalise_series_2d(Q)
    for i_t in range(X.shape[1] - L + 1):
        S = z_normalise_series_2d(X[:, i_t : i_t + L])
        assert_almost_equal(dist_profile[i_t], np.sum((S - Q) ** 2))
