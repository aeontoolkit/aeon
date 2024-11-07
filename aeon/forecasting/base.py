"""BaseForecaster class.

A simplified first base class for foreacasting models. The focus here is on a
specific form of forecasting: longer series, long winodws and single step forecasting.

aeon enhancement proposal
https://github.com/aeon-toolkit/aeon-admin/pull/14

"""

from abc import abstractmethod

import numpy as np
import pandas as pd

from aeon.base import BaseSeriesEstimator
from aeon.base._base_series import VALID_INPUT_TYPES


class BaseForecaster(BaseSeriesEstimator):
    """
    Abstract base class for time series forecasters.

    The base forecaster specifies the methods and method signatures that all
    forecasters have to implement. Attributes with an underscore suffix are set in the
    method fit.

    Parameters
    ----------
    horizon : int, default =1
        The number of time steps ahead to forecast. If horizon is one, the forecaster
        will learn to predict one point ahead.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "fit_is_empty": False,
        "y_inner_type": "np.ndarray",
    }

    def __init__(self, horizon=1, axis=1):
        self.horizon = horizon
        self.meta_ = None  # Meta data related to y on the last fit
        super().__init__(axis)

    def fit(self, y, exog=None):
        """Fit forecaster to series y.

        Fit a forecaster to predict self.horizon steps ahead using y.

        Parameters
        ----------
        y : np.ndarray
            A time series on which to learn a forecaster to predict horizon ahead
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y

        Returns
        -------
        self
            Fitted BaseForecaster.
        """
        # Validate y

        # Convert if necessary
        self._check_X(y, self.axis)
        y = self._convert_y(y, self.axis)
        if exog is not None:
            raise NotImplementedError("Exogenous variables not yet supported")
        # Validate exog
        self.is_fitted = True
        return self._fit(y, exog)

    @abstractmethod
    def _fit(self, y, exog=None): ...

    def predict(self, y=None, exog=None):
        """Predict the next horizon steps ahead.

        Parameters
        ----------
        y : np.ndarray, default = None
            A time series to predict the next horizon value for. If None,
            predict the next horizon value after series seen in fit.
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y

        Returns
        -------
        float
            single prediction self.horizon steps ahead of y.
        """
        if y is not None:
            self._check_X(y, self.axis)
            y = self._convert_y(y, self.axis)
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before predicting")
        if exog is not None:
            raise NotImplementedError("Exogenous variables not yet supported")
        return self._predict(y, exog)

    @abstractmethod
    def _predict(self, y=None, exog=None): ...

    def forecast(self, y, exog=None):
        """

        Forecast basically fit_predict.

        Returns
        -------
        np.ndarray
            single prediction directly after the last point in X.
        """
        self._check_X(y, self.axis)
        y = self._convert_y(y, self.axis)
        self._check_X(y, self.axis)
        y = self._convert_y(y, self.axis)
        return self._forecast(y, exog)

    def _forecast(self, y=None, exog=None):
        """Forecast values for time series X."""
        self.fit(y, exog)
        return self.predict(y, exog)

    def _convert_y(self, y: VALID_INPUT_TYPES, axis: int):
        """Convert y to self.get_tag("y_inner_type")."""
        if axis > 1 or axis < 0:
            raise ValueError(f"Input axis should be 0 or 1, saw {axis}")

        inner_type = self.get_tag("y_inner_type")
        if not isinstance(inner_type, list):
            inner_type = [inner_type]
        inner_names = [i.split(".")[-1] for i in inner_type]

        input = type(y).__name__
        if input not in inner_names:
            if inner_names[0] == "ndarray":
                y = y.to_numpy()
            elif inner_names[0] == "DataFrame":
                # converting a 1d array will create a 2d array in axis 0 format
                transpose = False
                if y.ndim == 1 and axis == 1:
                    transpose = True
                y = pd.DataFrame(y)
                if transpose:
                    y = y.T
            else:
                raise ValueError(
                    f"Unsupported inner type {inner_names[0]} derived from {inner_type}"
                )
        if y.ndim > 1 and self.axis != axis:
            y = y.T
        elif y.ndim == 1 and isinstance(y, np.ndarray):
            y = y[np.newaxis, :] if self.axis == 1 else y[:, np.newaxis]
        return y
