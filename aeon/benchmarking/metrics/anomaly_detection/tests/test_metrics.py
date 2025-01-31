"""Test cases for the range-based anomaly detection metrics."""

import numpy as np

from aeon.benchmarking.metrics.anomaly_detection.range_metrics import (
    ts_fscore,
    ts_precision,
    ts_recall,
)


def test_single_overlapping_range():
    """Test for single overlapping range."""
    y_pred = [(1, 4)]
    y_real = [(2, 6)]
    expected_precision = 0.750000
    expected_recall = 0.600000
    expected_f1 = 0.666667

    precision = ts_precision(y_pred, y_real, gamma="one", bias_type="flat")
    recall = ts_recall(
        y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0, udf_gamma=None
    )
    f1_score = ts_fscore(
        y_pred,
        y_real,
        gamma="one",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
        udf_gamma=None,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for single overlapping range! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for single overlapping range! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for single overlapping range! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )


def test_multiple_non_overlapping_ranges():
    """Test for multiple non-overlapping ranges."""
    y_pred = [(1, 2), (7, 8)]
    y_real = [(3, 4), (9, 10)]
    expected_precision = 0.000000
    expected_recall = 0.000000
    expected_f1 = 0.000000

    precision = ts_precision(y_pred, y_real, gamma="one", bias_type="flat")
    recall = ts_recall(
        y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0, udf_gamma=None
    )
    f1_score = ts_fscore(
        y_pred,
        y_real,
        gamma="one",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
        udf_gamma=None,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for multiple non-overlapping ranges! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for multiple non-overlapping ranges! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for multiple non-overlapping ranges! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )


def test_multiple_overlapping_ranges():
    """Test for multiple overlapping ranges."""
    y_pred = [(1, 3), (5, 7)]
    y_real = [(2, 6), (8, 10)]
    expected_precision = 0.666667
    expected_recall = 0.400000
    expected_f1 = 0.500000

    precision = ts_precision(y_pred, y_real, gamma="one", bias_type="flat")
    recall = ts_recall(
        y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0, udf_gamma=None
    )
    f1_score = ts_fscore(
        y_pred,
        y_real,
        gamma="one",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
        udf_gamma=None,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for multiple overlapping ranges! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for multiple overlapping ranges! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for multiple overlapping ranges! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )


def test_nested_lists_of_predictions():
    """Test for nested lists of predictions."""
    y_pred = [[(1, 3), (5, 7)], [(10, 12)]]
    y_real = [(2, 6), (8, 10)]
    expected_precision = 0.555556
    expected_recall = 0.566667
    expected_f1 = 0.561056

    precision = ts_precision(y_pred, y_real, gamma="one", bias_type="flat")
    recall = ts_recall(
        y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0, udf_gamma=None
    )
    f1_score = ts_fscore(
        y_pred,
        y_real,
        gamma="one",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
        udf_gamma=None,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for nested lists of predictions! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for nested lists of predictions! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for nested lists of predictions! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )


def test_all_encompassing_range():
    """Test for all encompassing range."""
    y_pred = [(1, 10)]
    y_real = [(2, 3), (5, 6), (8, 9)]
    expected_precision = 0.600000
    expected_recall = 1.000000
    expected_f1 = 0.750000

    precision = ts_precision(y_pred, y_real, gamma="one", bias_type="flat")
    recall = ts_recall(
        y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0, udf_gamma=None
    )
    f1_score = ts_fscore(
        y_pred,
        y_real,
        gamma="one",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
        udf_gamma=None,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for all encompassing range! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for all encompassing range! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for all encompassing range! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )


def test_binary_input_example():
    """Test for binary input sequences (existing example)."""
    y_pred_binary = np.array([0, 1, 1, 0])
    y_true_binary = np.array([0, 1, 0, 0])
    expected_precision = 0.500000
    expected_recall = 1.000000
    expected_f1 = 0.666667

    precision = ts_precision(
        y_pred_binary, y_true_binary, gamma="reciprocal", bias_type="flat"
    )
    recall = ts_recall(
        y_pred_binary,
        y_true_binary,
        gamma="reciprocal",
        bias_type="flat",
        alpha=0.0,
        udf_gamma=None,
    )
    f1_score = ts_fscore(
        y_pred_binary,
        y_true_binary,
        gamma="reciprocal",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
        udf_gamma=None,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for binary input example! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for binary input example! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for binary input example! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )


def test_multiple_overlapping_ranges_with_gamma_reciprocal():
    """Test for multiple overlapping ranges with gamma=reciprocal."""
    y_pred = [(1, 3), (5, 7)]
    y_real = [(2, 6), (8, 10)]
    expected_precision = 0.666667
    expected_recall = 0.200000
    expected_f1 = 0.307692

    precision = ts_precision(y_pred, y_real, gamma="reciprocal", bias_type="flat")
    recall = ts_recall(
        y_pred, y_real, gamma="reciprocal", bias_type="flat", alpha=0.0, udf_gamma=None
    )
    f1_score = ts_fscore(
        y_pred,
        y_real,
        gamma="reciprocal",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
        udf_gamma=None,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for multiple overlapping ranges! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for multiple overlapping ranges! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for multiple overlapping ranges! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )


def test_multiple_overlapping_ranges_with_bias_middle():
    """Test for multiple overlapping ranges with bias_type=middle."""
    y_pred = [(1, 3), (5, 7)]
    y_real = [(2, 6), (8, 10)]
    expected_precision = 0.750000
    expected_recall = 0.333333
    expected_f1 = 0.461538

    precision = ts_precision(y_pred, y_real, gamma="one", bias_type="middle")
    recall = ts_recall(
        y_pred, y_real, gamma="one", bias_type="middle", alpha=0.0, udf_gamma=None
    )
    f1_score = ts_fscore(
        y_pred,
        y_real,
        gamma="one",
        p_bias="middle",
        r_bias="middle",
        p_alpha=0.0,
        r_alpha=0.0,
        udf_gamma=None,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for multiple overlapping ranges! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for multiple overlapping ranges! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for multiple overlapping ranges! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )


def test_multiple_overlapping_ranges_with_bias_middle_gamma_reciprocal():
    """Test for multiple overlapping ranges with bias_type=middle, gamma=reciprocal."""
    y_pred = [(1, 3), (5, 7)]
    y_real = [(2, 6), (8, 10)]
    expected_precision = 0.750000
    expected_recall = 0.166667
    expected_f1 = 0.272727

    precision = ts_precision(y_pred, y_real, gamma="reciprocal", bias_type="middle")
    recall = ts_recall(
        y_pred,
        y_real,
        gamma="reciprocal",
        bias_type="middle",
        alpha=0.0,
        udf_gamma=None,
    )
    f1_score = ts_fscore(
        y_pred,
        y_real,
        gamma="reciprocal",
        p_bias="middle",
        r_bias="middle",
        p_alpha=0.0,
        r_alpha=0.0,
        udf_gamma=None,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for multiple overlapping ranges! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for multiple overlapping ranges! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for multiple overlapping ranges! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )
