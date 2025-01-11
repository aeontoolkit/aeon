"""Test cases for the range-based anomaly detection metrics."""

import numpy as np
import pytest

from aeon.benchmarking.metrics.anomaly_detection.range_metrics import (
    ts_fscore,
    ts_precision,
    ts_recall,
)


# Test cases for metrics
@pytest.mark.parametrize(
    "y_pred, y_real, expected_precision, expected_recall, expected_f1",
    [
        ([(1, 4)], [(2, 6)], 0.750000, 0.600000, 0.666667),  # Single Overlapping Range
        (
            [(1, 2), (7, 8)],
            [(3, 4), (9, 10)],
            0.000000,
            0.000000,
            0.000000,
        ),  # Multiple Non-Overlapping Ranges
        (
            [(1, 3), (5, 7)],
            [(2, 6), (8, 10)],
            0.666667,
            0.500000,
            0.571429,
        ),  # Multiple Overlapping Ranges
        (
            [[(1, 3), (5, 7)], [(10, 12)]],
            [(2, 6), (8, 10)],
            0.555556,
            0.625000,
            0.588235,
        ),  # Nested Lists of Predictions
        (
            [(1, 10)],
            [(2, 3), (5, 6), (8, 9)],
            0.600000,
            1.000000,
            0.750000,
        ),  # All Encompassing Range
        (
            [(1, 2)],
            [(1, 1)],
            0.5,
            1.000000,
            0.666667,
        ),  # Converted Binary to Range-Based(Existing example)
    ],
)
def test_metrics(y_pred, y_real, expected_precision, expected_recall, expected_f1):
    """Test the range-based anomaly detection metrics."""
    precision = ts_precision(y_pred, y_real, gamma="one", bias_type="flat")
    recall = ts_recall(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0)
    f1_score = ts_fscore(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0)

    # Use assertions with detailed error messages for debugging
    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=f"Precision failed! Expected={expected_precision}, Got={precision}",
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=f"Recall failed! Expected={expected_recall}, Got={recall}",
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=f"F1-Score failed! Expected={expected_f1}, Got={f1_score}",
    )
