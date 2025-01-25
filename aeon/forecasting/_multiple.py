"""Container for direct forecasting."""

import numpy as np

from aeon.forecasting.base import BaseForecaster


def direct_forecasting(
    forecaster: BaseForecaster, y: np.ndarray, steps_ahead: int, exog=None
):
    """Forecast steps_ahead points from X using forecaster.

    Parameters
    ----------
    forecaster : BaseForecaster class or object.
        Forecaster object with forecast method.
    y : 1D np.ndarray
        Time series to train forecasters on.
    steps_ahead : int
        Number of steps ahead to forecast.
    window_size : int
    Fits a different forecaster to each horizon

    Returns
    -------
    np.ndarray
        Length steps_ahead array of forecasts.
    """
    if not isinstance(forecaster, BaseForecaster):
        raise ValueError("Forecaster must be a BaseForecaster object.")
    preds = np.zeros(steps_ahead)
    for i in range(1, steps_ahead + 1):
        f = forecaster.__class__(horizon=i, **forecaster.params)
        preds[i - 1] = f.forecast(y, exog)
    return preds


def recursive_forecasting(
    forecaster: BaseForecaster, y: np.ndarray, steps_ahead: int, window=1, exog=None
):
    """Forecast steps_ahead points from X using forecaster."""
    if not isinstance(forecaster, BaseForecaster):
        raise ValueError("Forecaster must be a BaseForecaster object.")
    if hasattr(forecaster, "window"):
        forecaster = forecaster.__class__(horizon=1, **forecaster.params)
    forecaster.fit(y, exog)
    preds = np.zeros(steps_ahead)
    y = y[-window:]
    for i in range(1, steps_ahead + 1):
        preds[i - 1] = forecaster.predict(y)
        y[:-1] = y[1:]
        y[window - 1] = preds[i - 1]
    return preds
