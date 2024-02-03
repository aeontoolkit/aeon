"""Forecasting testing utils."""

__author__ = ["mloning"]
__all__ = [
    "make_forecasting_problem",
]

import numpy as np
import pandas as pd

from aeon.forecasting.base import ForecastingHorizon
from aeon.testing.utils.data_gen._series import make_series
from aeon.utils.validation.forecasting import check_fh


def _get_n_columns(tag):
    """Return the the number of columns to use in tests."""
    n_columns_list = []
    if tag in ["univariate", "both"]:
        n_columns_list = [1, 2]
    elif tag == "multivariate":
        n_columns_list = [2]
    else:
        raise ValueError(f"Unexpected tag {tag} in _get_n_columns.")
    return n_columns_list


def make_forecasting_problem(
    n_timepoints=50,
    all_positive=True,
    index_type=None,
    make_X=False,
    n_columns=1,
    random_state=None,
):
    """Return test data for forecasting tests.

    Parameters
    ----------
    n_timepoints : int, optional
        Lenght of data, by default 50
    all_positive : bool, optional
        Only positive values or not, by default True
    index_type : e.g. pd.PeriodIndex, optional
        pandas Index type, by default None
    make_X : bool, optional
        Should X data also be returned, by default False
    n_columns : int, optional
        Number of columns of y, by default 1
    random_state : inst, str, float, optional
        Set seed of random state, by default None

    Returns
    -------
    ps.Series, pd.DataFrame
        y, if not make_X
        y, X if make_X
    """
    y = make_series(
        n_timepoints=n_timepoints,
        n_columns=n_columns,
        all_positive=all_positive,
        index_type=index_type,
        random_state=random_state,
    )

    if not make_X:
        return y

    X = make_series(
        n_timepoints=n_timepoints,
        n_columns=2,
        all_positive=all_positive,
        index_type=index_type,
        random_state=random_state,
    )
    X.index = y.index
    return y, X


def _assert_correct_pred_time_index(y_pred_index, cutoff, fh):
    assert isinstance(y_pred_index, pd.Index)
    fh = check_fh(fh)
    expected = fh.to_absolute(cutoff).to_pandas()
    y_pred_index.equals(expected)


def _assert_correct_columns(y_pred, y_train):
    """Check that forecast object has right column names."""
    if isinstance(y_pred, pd.DataFrame) and isinstance(y_train, pd.DataFrame):
        msg = (
            "forecast pd.DataFrame must have same column index as past data, "
            f"expected {y_train.columns} but found {y_pred.columns}"
        )
        assert (y_pred.columns == y_train.columns).all(), msg

    if isinstance(y_pred, pd.Series) and isinstance(y_train, pd.Series):
        msg = (
            "forecast pd.Series must have same name as past data, "
            f"expected {y_train.name} but found {y_pred.name}"
        )
        assert y_pred.name == y_train.name, msg


def _make_fh(cutoff, steps, fh_type, is_relative):
    """Construct forecasting horizons for testing."""
    from aeon.forecasting.tests import INDEX_TYPE_LOOKUP

    fh_class = INDEX_TYPE_LOOKUP[fh_type]

    if isinstance(steps, (int, np.integer)):
        steps = np.array([steps], dtype=int)

    elif isinstance(steps, pd.Timedelta):
        steps = [steps]

    if is_relative:
        return ForecastingHorizon(fh_class(steps), is_relative=is_relative)

    else:
        kwargs = {}

        if fh_type in ["datetime", "period"]:
            cutoff_freq = cutoff.freq
        if isinstance(cutoff, pd.Index):
            cutoff = cutoff[0]

        if fh_type == "datetime":
            steps *= cutoff_freq

        if fh_type == "period":
            kwargs = {"freq": cutoff_freq}

        values = cutoff + steps
        return ForecastingHorizon(fh_class(values, **kwargs), is_relative)
