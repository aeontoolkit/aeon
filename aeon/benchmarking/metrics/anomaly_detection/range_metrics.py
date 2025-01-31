"""Calculate Precision, Recall, and F1-Score for time series anomaly detection."""

__maintainer__ = []
__all__ = ["ts_precision", "ts_recall", "ts_fscore"]

import numpy as np


def _flatten_ranges(ranges):
    """
    If the input is a list of lists, it flattens it into a single list.

    Parameters
    ----------
    ranges : list of tuples or list of lists of tuples
        The ranges to flatten. each tuple shoulod be in the format of (start, end).

    Returns
    -------
    list of tuples
        A flattened list of ranges.

    Examples
    --------
    >>> _flatten_ranges([[(1, 5), (10, 15)], [(20, 25)]])
    [(1, 5), (10, 15), (20, 25)]
    """
    if not ranges:
        return []
    if isinstance(ranges[0], list):
        flat = []
        for sublist in ranges:
            for pred in sublist:
                flat.append(pred)
        return flat
    return ranges


def _calculate_bias(position, length, bias_type="flat"):
    """Calculate bias value based on position and length.

    Parameters
    ----------
    position : int
        Current position in the range
    length : int
        Total length of the range
    bias_type : str, default="flat"
        Type of bias to apply, Should be one of ["flat", "front", "middle", "back"].
    """
    if bias_type == "flat":
        return 1.0
    elif bias_type == "front":
        return 1.0 - (position - 1) / length
    elif bias_type == "middle":
        if length / 2 == 0:
            return 1.0
        if position <= length / 2:
            return position / (length / 2)
        else:
            return (length - position + 1) / (length / 2)
    elif bias_type == "back":
        return position / length
    else:
        raise ValueError(f"Invalid bias type: {bias_type}")


def _gamma_select(cardinality, gamma, udf_gamma=None):
    """Select a gamma value based on the cardinality type.

    Parameters
    ----------
    cardinality : int
        The number of overlapping ranges.
    gamma : str
        Gamma to use. Should be one of ["one", "reciprocal", "udf_gamma"].
    udf_gamma : float or None, optional
        The user-defined gamma value to use when `gamma` is set to "udf_gamma".
        Required if `gamma` is "udf_gamma".

    Returns
    -------
    float
        The selected gamma value.

    Raises
    ------
    ValueError
        If an invalid `gamma` type is provided or if `udf_gamma` is required
        but not provided.
    """
    if gamma == "one":
        return 1.0
    elif gamma == "reciprocal":
        return 1 / cardinality if cardinality > 1 else 1.0
    elif gamma == "udf_gamma":
        if udf_gamma is not None:
            return 1.0 / udf_gamma
        else:
            raise ValueError("udf_gamma must be provided for 'udf_gamma' gamma type.")
    else:
        raise ValueError(
            "Invalid gamma type. Choose from ['one', 'reciprocal', 'udf_gamma']."
        )


def _calculate_overlap_reward_precision(pred_range, overlap_set, bias_type):
    """Overlap Reward for y_pred.

    Parameters
    ----------
    pred_range : tuple
        The predicted range.
    overlap_set : set
        The set of overlapping positions.
    bias_type : str
        Type of bias to apply, Should be one of ["flat", "front", "middle", "back"].

    Returns
    -------
    float
        The weighted value for overlapping positions only.
    """
    start, end = pred_range
    length = end - start + 1

    max_value = 0  # Total possible weighted value for all positions.
    my_value = 0  # Weighted value for overlapping positions only.

    for i in range(1, length + 1):
        global_position = start + i - 1
        bias_value = _calculate_bias(i, length, bias_type)
        max_value += bias_value

        if global_position in overlap_set:
            my_value += bias_value

    return my_value / max_value if max_value > 0 else 0.0


def _calculate_overlap_reward_recall(real_range, overlap_set, bias_type):
    """Overlap Reward for y_real.

    Parameters
    ----------
    real_range : tuple
        The real range.
    overlap_set : set
        The set of overlapping positions.
    bias_type : str
        Type of bias to apply, Should be one of ["flat", "front", "middle", "back"].

    Returns
    -------
    float
        The weighted value for overlapping positions only.
    """
    start, end = real_range
    length = end - start + 1

    max_value = 0.0  # Total possible weighted value for all positions.
    my_value = 0.0  # Weighted value for overlapping positions only.

    for i in range(1, length + 1):
        global_position = start + i - 1
        bias_value = _calculate_bias(i, length, bias_type)
        max_value += bias_value

        if global_position in overlap_set:
            my_value += bias_value

    return my_value / max_value if max_value > 0 else 0.0


def _binary_to_ranges(binary_sequence):
    """
    Convert a binary sequence to a list of anomaly ranges.

    Parameters
    ----------
    binary_sequence : list
        Binary sequence where 1 indicates anomaly and 0 indicates normal.

    Returns
    -------
    list of tuples
        List of anomaly ranges as (start, end) tuples.

    """
    ranges = []
    start = None

    for i, val in enumerate(binary_sequence):
        if val and start is None:
            start = i
        elif not val and start is not None:
            ranges.append((start, i - 1))
            start = None

    if start is not None:
        ranges.append((start, len(binary_sequence) - 1))

    return ranges


def ts_precision(y_pred, y_real, gamma="one", bias_type="flat"):
    """
    Calculate Precision for time series anomaly detection.

    Precision measures the proportion of correctly predicted anomaly positions
    out of all all the predicted anomaly positions, aggregated across the entire time
    series.

    Parameters
    ----------
    y_pred : list of tuples
        The predicted anomaly ranges.
        - Each tuple represents a range (start, end) of the anomaly where
          start is starting index (inclusive) and end is ending index (inclusive).
    y_real : list of tuples or list of lists of tuples
        The real/actual (ground truth) ranges.
        - Each tuple represents a range (start, end) of the anomaly where
          start is starting index (inclusive) and end is ending index (inclusive).
        - If y_real is in the format of list of lists, they will be flattened into a
          single list of tuples bringing it to the above format.
    bias_type : str, default="flat"
        Type of bias to apply. Should be one of ["flat", "front", "middle", "back"].
    gamma : str, default="one"
        Cardinality type. Should be one of ["reciprocal", "one"].

    Returns
    -------
    float
        Precision

    Raises
    ------
    ValueError
        If an invalid `gamma` type is provided.
    ValueError
        If input sequence is binary and y_real and y_pred are of different lengths.

    References
    ----------
    .. [1] Tatbul, Nesime, Tae Jun Lee, Stan Zdonik, Mejbah Alam,and Justin Gottschlich.
       "Precision and Recall for Time Series." 32nd Conference on Neural Information
       Processing Systems (NeurIPS 2018), Montréal, Canada.
       http://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf
    """
    # Check if inputs are binary or range-based
    is_binary = False
    if isinstance(y_pred, (list, tuple, np.ndarray)) and isinstance(
        y_pred[0], (int, np.integer)
    ):
        is_binary = True
    elif isinstance(y_real, (list, tuple, np.ndarray)) and isinstance(
        y_real[0], (int, np.integer)
    ):
        is_binary = True

    if is_binary:
        if not isinstance(y_pred, (list, tuple, np.ndarray)) or not isinstance(
            y_real, (list, tuple, np.ndarray)
        ):
            raise ValueError(
                "For binary inputs, y_pred and y_real should be list or tuple, "
                "or numpy array of integers."
            )
        if len(y_pred) != len(y_real):
            raise ValueError(
                "For binary inputs, y_pred and y_real must be of the same length."
            )

        y_pred_ranges = _binary_to_ranges(y_pred)
        y_real_ranges = _binary_to_ranges(y_real)
    else:
        y_pred_ranges = y_pred
        y_real_ranges = y_real

    if gamma not in ["reciprocal", "one"]:
        raise ValueError("Invalid gamma type for precision. Use 'reciprocal' or 'one'.")

    # Flattening y_pred and y_real to resolve nested lists
    flat_y_pred = _flatten_ranges(y_pred_ranges)
    flat_y_real = _flatten_ranges(y_real_ranges)

    total_overlap_reward = 0.0
    total_cardinality = 0

    for pred_range in flat_y_pred:
        overlap_set = set()
        cardinality = 0

        for real_start, real_end in flat_y_real:
            overlap_start = max(pred_range[0], real_start)
            overlap_end = min(pred_range[1], real_end)

            if overlap_start <= overlap_end:
                overlap_set.update(range(overlap_start, overlap_end + 1))
                cardinality += 1

        overlap_reward = _calculate_overlap_reward_precision(
            pred_range, overlap_set, bias_type
        )
        gamma_value = _gamma_select(cardinality, gamma)
        total_overlap_reward += gamma_value * overlap_reward
        total_cardinality += 1

    precision = (
        total_overlap_reward / total_cardinality if total_cardinality > 0 else 0.0
    )
    return precision


def ts_recall(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0, udf_gamma=None):
    """
    Calculate Recall for time series anomaly detection.

    Recall measures the proportion of correctly predicted anomaly positions
    out of all the real/actual (ground truth) anomaly positions, aggregated across the
    entire time series.

    Parameters
    ----------
    y_pred : list of tuples
        The predicted anomaly ranges.
        - Each tuple represents a range (start, end) of the anomaly where
          start is starting index (inclusive) and end is ending index (inclusive).
    y_real : list of tuples or list of lists of tuples
        The real/actual (ground truth) ranges.
        - Each tuple represents a range (start, end) of the anomaly where
          start is starting index (inclusive) and end is ending index (inclusive).
        - If y_real is in the format of list of lists, they will be flattened into a
          single list of tuples bringing it to the above format.
    gamma : str, default="one"
        Cardinality type. Should be one of ["reciprocal", "one", "udf_gamma"].
    bias_type : str, default="flat"
        Type of bias to apply. Should be one of ["flat", "front", "middle", "back"].
    alpha : float, default: 0.0
        Weight for existence reward in recall calculation.
    udf_gamma : int or None, default=None
        User-defined gamma value.

    Returns
    -------
    float
        Recall

    Raises
    ------
    ValueError
        If input sequence is binary and y_real and y_pred are of different lengths.

    References
    ----------
    .. [1] Tatbul, Nesime, Tae Jun Lee, Stan Zdonik, Mejbah Alam,and Justin Gottschlich.
       "Precision and Recall for Time Series." 32nd Conference on Neural Information
       Processing Systems (NeurIPS 2018), Montréal, Canada.
       http://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf
    """
    is_binary = False
    if isinstance(y_pred, (list, tuple, np.ndarray)) and isinstance(
        y_pred[0], (int, np.integer)
    ):
        is_binary = True
    elif isinstance(y_real, (list, tuple, np.ndarray)) and isinstance(
        y_real[0], (int, np.integer)
    ):
        is_binary = True

    if is_binary:
        if not isinstance(y_pred, (list, tuple, np.ndarray)) or not isinstance(
            y_real, (list, tuple, np.ndarray)
        ):
            raise ValueError(
                "For binary inputs, y_pred and y_real should be list or tuple, "
                "or numpy array of integers."
            )
        if len(y_pred) != len(y_real):
            raise ValueError(
                "For binary inputs, y_pred and y_real must be of the same length."
            )

        y_pred_ranges = _binary_to_ranges(y_pred)
        y_real_ranges = _binary_to_ranges(y_real)
    else:
        y_pred_ranges = y_pred
        y_real_ranges = y_real

    # Flattening y_pred and y_real to resolve nested lists
    flat_y_pred = _flatten_ranges(y_pred_ranges)
    flat_y_real = _flatten_ranges(y_real_ranges)

    total_overlap_reward = 0.0

    for real_range in flat_y_real:
        overlap_set = set()
        cardinality = 0

        for pred_range in flat_y_pred:
            overlap_start = max(real_range[0], pred_range[0])
            overlap_end = min(real_range[1], pred_range[1])

            if overlap_start <= overlap_end:
                overlap_set.update(range(overlap_start, overlap_end + 1))
                cardinality += 1

        existence_reward = 1.0 if overlap_set else 0.0

        if overlap_set:
            overlap_reward = _calculate_overlap_reward_recall(
                real_range, overlap_set, bias_type
            )
            gamma_value = _gamma_select(cardinality, gamma, udf_gamma)
            overlap_reward *= gamma_value
        else:
            overlap_reward = 0.0

        recall_score = alpha * existence_reward + (1 - alpha) * overlap_reward
        total_overlap_reward += recall_score

    recall = total_overlap_reward / len(flat_y_real) if flat_y_real else 0.0
    return recall


def ts_fscore(
    y_pred,
    y_real,
    gamma="one",
    p_bias="flat",
    r_bias="flat",
    p_alpha=0.0,
    r_alpha=0.0,
    udf_gamma=None,
):
    """
    Calculate F1-Score for time series anomaly detection.

    F-1 Score is the harmonic mean of Precision and Recall, providing
    a single metric to evaluate the performance of an anomaly detection model.

    Parameters
    ----------
    y_pred : list of tuples
        The predicted anomaly ranges.
        - Each tuple represents a range (start, end) of the anomaly where
          start is starting index (inclusive) and end is ending index (inclusive).
    y_real : list of tuples or list of lists of tuples
        The real/actual (ground truth) ranges.
        - Each tuple represents a range (start, end) of the anomaly where
          start is starting index (inclusive) and end is ending index (inclusive).
        - If y_real is in the format of list of lists, they will be flattened into a
          single list of tuples bringing it to the above format.
    gamma : str, default="one"
        Cardinality type. Should be one of ["reciprocal", "one", "udf_gamma"].
    p_bias : str, default="flat"
        Type of bias to apply for precision.
        Should be one of ["flat", "front", "middle", "back"].
    r_bias : str, default="flat"
        Type of bias to apply for recall.
        Should be one of ["flat", "front", "middle", "back"].
    p_alpha : float, default=0.0
        Weight for existence reward in Precision calculation.
    r_alpha : float, default=0.0
        Weight for existence reward in Recall calculation.
    udf_gamma : int or None, default=None
        User-defined gamma value.

    Returns
    -------
    float
        F1-Score

    References
    ----------
    .. [1] Tatbul, Nesime, Tae Jun Lee, Stan Zdonik, Mejbah Alam,and Justin Gottschlich.
       "Precision and Recall for Time Series." 32nd Conference on Neural Information
       Processing Systems (NeurIPS 2018), Montréal, Canada.
       http://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf
    """
    precision = ts_precision(y_pred, y_real, gamma, p_bias)
    recall = ts_recall(y_pred, y_real, gamma, r_bias, r_alpha, udf_gamma)

    if precision + recall > 0:
        fscore = 2 * (precision * recall) / (precision + recall)
    else:
        fscore = 0.0

    return fscore
