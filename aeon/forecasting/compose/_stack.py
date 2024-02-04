"""Implements forecasters for combining forecasts via stacking."""

__author__ = ["mloning", "fkiraly", "indinewton"]
__all__ = ["StackingForecaster"]

from warnings import warn

import numpy as np
import pandas as pd

from aeon.forecasting.base._meta import _HeterogenousEnsembleForecaster
from aeon.forecasting.model_selection import SingleWindowSplitter
from aeon.utils.validation.forecasting import check_regressor


class StackingForecaster(_HeterogenousEnsembleForecaster):
    """StackingForecaster.

    Stacks two or more Forecasters and uses a meta-model (regressor) to infer
    the final predictions from the predictions of the given forecasters.

    Parameters
    ----------
    forecasters : list of (str, estimator) tuples
        Estimators to apply to the input series.
    regressor: sklearn-like regressor, optional, default=None.
        The regressor is used as a meta-model and trained with the predictions
        of the ensemble forecasters as exog data and with y as endog data. The
        length of the data is dependent to the given fh. If None, then
        a GradientBoostingRegressor(max_depth=5) is used.
        The regressor can also be a sklearn.Pipeline().
    random_state : int, RandomState instance or None, default=None
        Used to set random_state of the default regressor.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for fit. None means 1 unless
        in a joblib.parallel_backend context.
        -1 means using all processors.

    Attributes
    ----------
    regressor_ : sklearn-like regressor
        Fitted meta-model (regressor)

    Examples
    --------
    >>> from aeon.forecasting.compose import StackingForecaster
    >>> from aeon.forecasting.naive import NaiveForecaster
    >>> from aeon.forecasting.trend import PolynomialTrendForecaster
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecasters = [
    ...     ("trend", PolynomialTrendForecaster()),
    ...     ("naive", NaiveForecaster()),
    ... ]
    >>> forecaster = StackingForecaster(forecasters=forecasters)
    >>> forecaster.fit(y=y, fh=[1,2,3])
    StackingForecaster(...)
    >>> y_pred = forecaster.predict()
    """

    _tags = {
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": True,
        "capability:missing_values": True,
        "y_input_type": "univariate",
        "X-y-must-have-same-index": True,
    }

    def __init__(self, forecasters, regressor=None, random_state=None, n_jobs=None):
        super().__init__(forecasters=forecasters, n_jobs=n_jobs)
        self.regressor = regressor
        self.random_state = random_state

        self._anytagis_then_set("ignores-exogeneous-X", False, True, forecasters)
        self._anytagis_then_set("capability:missing_values", False, True, forecasters)
        self._anytagis_then_set("fit_is_empty", False, True, forecasters)

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        _, forecasters = self._check_forecasters()
        self.regressor_ = check_regressor(
            regressor=self.regressor, random_state=self.random_state
        )

        # split training series into training set to fit forecasters and
        # validation set to fit meta-learner
        inner_fh = fh.to_relative(self.cutoff)
        cv = SingleWindowSplitter(fh=inner_fh)
        train_window, test_window = next(cv.split(y))
        y_train = y.iloc[train_window]
        y_test = y.iloc[test_window]
        if X is not None:
            X_test = X.iloc[test_window]
            X_train = X.iloc[train_window]
        else:
            X_test = None
            X_train = None

        # fit forecasters on training window
        self._fit_forecasters(forecasters, y_train, fh=inner_fh, X=X_train)
        y_preds = self._predict_forecasters(fh=inner_fh, X=X_test)

        y_meta = y_test.values
        X_meta = np.column_stack(y_preds)

        # fit final regressor on on validation window
        self.regressor_.fit(X_meta, y_meta)

        # refit forecasters on entire training series
        self._fit_forecasters(forecasters, y, fh=fh, X=X)

        return self

    def _update(self, y, X=None, update_params=True):
        """Update fitted parameters.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame
        update_params : bool, optional (default=True)

        Returns
        -------
        self : an instance of self
        """
        if update_params:
            warn("Updating `final regressor is not implemented")
        for forecaster in self.forecasters_:
            forecaster.update(y, X, update_params=update_params)
        return self

    def _predict(self, fh=None, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        y_preds = np.column_stack(self._predict_forecasters(fh=fh, X=X))
        y_pred = self.regressor_.predict(y_preds)
        # index = y_preds.index
        index = self.fh.to_absolute(self.cutoff).to_pandas()
        return pd.Series(y_pred, index=index)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict
        """
        from aeon.forecasting.naive import NaiveForecaster

        f1 = NaiveForecaster()
        f2 = NaiveForecaster(strategy="mean", window_length=3)
        params = {"forecasters": [("f1", f1), ("f2", f2)]}

        return params
