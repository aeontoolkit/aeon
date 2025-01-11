"""Tests for the IDK Class."""

import numpy as np

from aeon.anomaly_detection import IDK


def test_idk_univariate():
    """Test IDK on univariate data."""
    rng = np.random.default_rng(seed=2)
    series = rng.normal(size=(100,))
    series[50:58] -= 5

    ad = IDK(psi1=8, psi2=2, width=1, random_state=2)
    pred = ad.fit_predict(series)
    ad_sliding = IDK(psi1=16, psi2=4, width=10, sliding=True, random_state=1)
    pred_sliding = ad_sliding.fit_predict(series)

    assert pred.shape == (100,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 58
    assert pred_sliding.shape == (91,)
    assert pred_sliding.dtype == np.float64
    assert 60 <= np.argmax(pred_sliding) <= 80
