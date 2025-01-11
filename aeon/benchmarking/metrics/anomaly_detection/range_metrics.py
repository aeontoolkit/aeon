"""Calculate Precision, Recall, and F1-Score for time series anomaly detection."""

maintainer = []
__all__ = ["ts_precision", "ts_recall", "ts_fscore"]


def flatten_ranges(ranges):
    """
    If the input is a list of lists, it flattens it into a single list.

    Parameters
    ----------
    ranges : list of tuples or list of lists of tuples
        The ranges to flatten.

    Returns
    -------
    list of tuples
        A flattened list of ranges.
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


def calculate_bias(position, length, bias_type="flat"):
    """Calculate bias value based on position and length.

    Parameters
    ----------
    position : int
        Current position in the range
    length : int
        Total length of the range
    bias_type : str
        Type of bias to apply, Should be one of ["flat", "front", "middle", "back"].
        (default: "flat")
    """
    if bias_type == "flat":
        return 1.0
    elif bias_type == "front":
        return 1.0 - (position - 1) / length
    elif bias_type == "middle":
        return 1.0 - abs(2 * (position - 1) / (length - 1) - 1) if length > 1 else 1.0
    elif bias_type == "back":
        return position / length
    else:
        raise ValueError(f"Invalid bias type: {bias_type}")


def gamma_select(cardinality, gamma, udf_gamma=None):
    """Select a gamma value based on the cardinality type."""
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
        raise ValueError("Invalid gamma type.")


def ts_precision(y_pred, y_real, gamma="one", bias_type="flat", udf_gamma=None):
    """
    Calculate Global Precision for time series anomaly detection.

    Parameters
    ----------
    y_pred : list of tuples or list of lists of tuples
        The predicted ranges.
    y_real : list of tuples or list of lists of tuples
        The real (actual) ranges.
    gamma : str
        Cardinality type. Should be one of ["reciprocal", "one", "udf_gamma"].
        (default: "one")
    bias_type : str
        Type of bias to apply. Should be one of ["flat", "front", "middle", "back"].
        (default: "flat")
    udf_gamma : int or None
        User-defined gamma value. (default: None)

    Returns
    -------
    float
        Global Precision
    """
    # Flattening y_pred and y_real to resolve nested lists
    flat_y_pred = flatten_ranges(y_pred)
    flat_y_real = flatten_ranges(y_real)

    overlapping_weighted_positions = 0.0
    total_pred_weight = 0.0

    for pred_range in flat_y_pred:
        start_pred, end_pred = pred_range
        length_pred = end_pred - start_pred + 1

        for i in range(1, length_pred + 1):
            pos = start_pred + i - 1
            bias = calculate_bias(i, length_pred, bias_type)

            # Check if the position is in any real range
            in_real = any(
                real_start <= pos <= real_end for real_start, real_end in flat_y_real
            )

            if in_real:
                gamma_value = gamma_select(1, gamma, udf_gamma)
                overlapping_weighted_positions += bias * gamma_value

            total_pred_weight += bias

    precision = (
        overlapping_weighted_positions / total_pred_weight
        if total_pred_weight > 0
        else 0.0
    )
    return precision


def ts_recall(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0, udf_gamma=None):
    """
    Calculate Global Recall for time series anomaly detection.

    Parameters
    ----------
    y_pred : list of tuples or list of lists of tuples
        The predicted ranges.
    y_real : list of tuples or list of lists of tuples
        The real (actual) ranges.
    gamma : str
        Cardinality type. Should be one of ["reciprocal", "one", "udf_gamma"].
        (default: "one")
    bias_type : str
        Type of bias to apply. Should be one of ["flat", "front", "middle", "back"].
        (default: "flat")
    alpha : float
        Weight for existence reward in recall calculation. (default: 0.0)
    udf_gamma : int or None
        User-defined gamma value. (default: None)

    Returns
    -------
    float
        Global Recall
    """
    # Flattening y_pred and y_real
    flat_y_pred = flatten_ranges(y_pred)
    flat_y_real = flatten_ranges(y_real)

    overlapping_weighted_positions = 0.0
    total_real_weight = 0.0

    for real_range in flat_y_real:
        start_real, end_real = real_range
        length_real = end_real - start_real + 1

        for i in range(1, length_real + 1):
            pos = start_real + i - 1
            bias = calculate_bias(i, length_real, bias_type)

            # Check if the position is in any predicted range
            in_pred = any(
                pred_start <= pos <= pred_end for pred_start, pred_end in flat_y_pred
            )

            if in_pred:
                gamma_value = gamma_select(1, gamma, udf_gamma)
                overlapping_weighted_positions += bias * gamma_value

            total_real_weight += bias

    recall = (
        overlapping_weighted_positions / total_real_weight
        if total_real_weight > 0
        else 0.0
    )
    return recall


def ts_fscore(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0, udf_gamma=None):
    """
    Calculate F1-Score for time series anomaly detection.

    Parameters
    ----------
    y_pred : list of tuples or list of lists of tuples
        The predicted ranges.
    y_real : list of tuples or list of lists of tuples
        The real (actual) ranges.
    gamma : str
        Cardinality type. Should be one of ["reciprocal", "one", "udf_gamma"].
        (default: "one")
    bias_type : str
        Type of bias to apply. Should be one of ["flat", "front", "middle", "back"].
        (default: "flat")
    udf_gamma : int or None
        User-defined gamma value. (default: None)

    Returns
    -------
    float
        F1-Score
    """
    precision = ts_precision(y_pred, y_real, gamma, bias_type, udf_gamma=udf_gamma)
    recall = ts_recall(y_pred, y_real, gamma, bias_type, alpha, udf_gamma=udf_gamma)

    if precision + recall > 0:
        fscore = 2 * (precision * recall) / (precision + recall)
    else:
        fscore = 0.0

    return fscore
