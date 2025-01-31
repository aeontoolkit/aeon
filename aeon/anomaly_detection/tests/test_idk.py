"""Tests for the IDK Class."""

import numpy as np
from numpy.testing import assert_allclose

from aeon.anomaly_detection import IDK


def test_idk_univariate_basic():
    """Test IDK on basic univariate data."""
    rng = np.random.default_rng(seed=2)
    series = rng.normal(size=(100,))
    series[50:58] -= 5

    ad = IDK(psi1=8, psi2=2, width=1, random_state=2)
    pred = ad.fit_predict(series)

    assert pred.shape == (100,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 58


def test_idk_univariate_sliding():
    """Test IDK with sliding window on univariate data."""
    rng = np.random.default_rng(seed=2)
    series = rng.normal(size=(100,))
    series[50:58] -= 5

    ad_sliding = IDK(psi1=16, psi2=4, width=10, sliding=True, random_state=1)
    pred_sliding = ad_sliding.fit_predict(series)

    assert pred_sliding.shape == (100,)
    assert pred_sliding.dtype == np.float64
    assert 60 <= np.argmax(pred_sliding) <= 80


def test_idk_univariate_custom_series():
    """Test IDK on a custom univariate series with assert_allclose."""
    series1 = np.array(
        [
            0.18905338,
            -0.52274844,
            -0.41306354,
            -2.44146738,
            1.79970738,
            1.14416587,
            -0.32542284,
            0.77380659,
            0.28121067,
            -0.55382284,
        ]
    )
    expected = [0.52333333, 0.19, 0.52333333]

    ad_2 = IDK(psi1=4, psi2=2, width=3, t=10, random_state=2)
    pred2 = ad_2.fit_predict(series1)

    assert pred2.shape == (3,)
    assert pred2.dtype == np.float64
    assert_allclose(pred2, expected, atol=0.01)
