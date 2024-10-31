"""Window-based regression forecaster.

General purpose forecaster to use with any scikit learn or aeon compatible
regressor. Simply forms a collection of windows from the time series and trains to
predict the next
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from aeon.forecasting.base import BaseForecaster


class RegressionForecaster(BaseForecaster):
    def __init__(self, window, horizon=1, regressor=None):
        self.regressor = regressor
        super().__init__(horizon, window)

    def _fit(self, y, exog=None):
        """Fit forecaster to time series.

        Split X into windows of length window and train the forecaster on each window
        to predict the horizon ahead.

        Parameters
        ----------
        X : Time series on which to learn a forecaster

        Returns
        -------
        self
            Fitted estimator
        """
        # Window data
        if self.regressor is None:
            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = self.regressor
        y = y.squeeze()
        X = np.lib.stride_tricks.sliding_window_view(y, window_shape=self.window)
        # Ignore the final horizon values: need to store these for pred with empty y
        X = X[: -self.horizon]
        # Extract y
        y = y[self.window + self.horizon - 1 :]
        self.last_ = y[-self.window :]
        self.last_ = self.last_.reshape(1, -1)
        self.regressor_.fit(X=X, y=y)
        return self

    def _predict(self, y=None, exog=None):
        """Predict values for time series X."""
        if y is None:
            return self.regressor_.predict(self.last_)
        last = y[:, -self.window :]
        return self.regressor_.predict(last)

    def _forecast(self, y, exog=None):
        """Forecast values for time series X.

        NOTE: deal with horizons
        """
        self.fit(y, exog)
        return self.predict(y)

    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default='default'
            Name of the parameter set to return.

        Returns
        -------
        dict
            Dictionary of testing parameter settings.
        """
        return {"window": 4}
