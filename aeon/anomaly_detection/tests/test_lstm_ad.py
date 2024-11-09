"""Tests for the LSTM_AD class."""

import numpy as np
import pytest

from aeon.anomaly_detection.deep_learning import LSTM_AD
from aeon.testing.data_generation._legacy import make_series
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_lstmad_univariate():
    """Test LSTM_AD univariate output."""
    series = make_series(n_timepoints=1000, return_numpy=True, random_state=42)
    labels = np.zeros(1000).astype(int)

    # Create anomalies
    anomaly_indices = np.random.choice(1000, 20, replace=False)
    series[anomaly_indices] += np.random.normal(loc=0, scale=4, size=(20,))
    labels[anomaly_indices] = 1

    ad = LSTM_AD(
        n_layers=4, n_nodes=16, window_size=10, prediction_horizon=1, n_epochs=1
    )
    ad.fit(series, labels, axis=0)
    pred = ad.predict(series, axis=0)

    assert pred.shape == (1000,)
    assert pred.dtype == np.int_


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_lstmad_multivariate():
    """Test LSTM_AD multivariate output."""
    series = make_series(
        n_timepoints=1000, n_columns=3, return_numpy=True, random_state=42
    )
    labels = np.zeros(1000).astype(int)

    # Create anomalies
    anomaly_indices = np.random.choice(1000, 50, replace=False)
    series[anomaly_indices] += np.random.normal(loc=0, scale=4, size=(50, 3))
    labels[anomaly_indices] = 1

    ad = LSTM_AD(
        n_layers=4, n_nodes=16, window_size=10, prediction_horizon=1, n_epochs=1
    )
    ad.fit(series, labels, axis=0)
    pred = ad.predict(series, axis=0)

    assert pred.shape == (1000,)
    assert pred.dtype == np.int_
