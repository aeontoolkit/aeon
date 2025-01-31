"""
Abstract base class for time series regressors.

    class name: BaseRegressor

Defining methods:
    fitting         - fit(self, X, y)
    predicting      - predict(self, X)

Inherited inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
"""

__maintainer__ = []
__all__ = ["BaseRegressor"]

from abc import abstractmethod
from typing import final

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import get_scorer, get_scorer_names
from sklearn.model_selection import cross_val_predict
from sklearn.utils.multiclass import type_of_target

from aeon.base import BaseCollectionEstimator
from aeon.base._base import _clone_estimator


class BaseRegressor(RegressorMixin, BaseCollectionEstimator):
    """Abstract base class for time series regressors.

    The base regressor specifies the methods and method signatures that all
    regressors have to implement. Attributes with a underscore suffix are set in the
    method fit.

    Attributes
    ----------
    _estimator_type : string
        The type of estimator. Required by some ``sklearn`` tools, set to "regressor".
    """

    _tags = {
        "fit_is_empty": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
    }

    @abstractmethod
    def __init__(self):
        super().__init__()

    @final
    def fit(self, X, y) -> BaseCollectionEstimator:
        """Fit time series regressor to training data.

        Parameters
        ----------
        X : np.ndarray or list
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. Other types are
            allowed and converted into one of the above.

            Different estimators have different capabilities to handle different
            types of input. If ``self.get_tag("capability:multivariate")`` is False,
            they cannot handle multivariate series, so either ``n_channels == 1`` is
            true or X is 2D of shape ``(n_cases, n_timepoints)``. If ``self.get_tag(
            "capability:unequal_length")`` is False, they cannot handle unequal
            length input. In both situations, a ``ValueError`` is raised if X has a
            characteristic that the estimator does not have the capability for is
            passed.
        y : np.ndarray
            1D np.array of float, of shape ``(n_cases)`` - regression targets
            (ground truth) for fitting indices corresponding to instance indices in X.

        Returns
        -------
        self : BaseRegressor
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        X, y = self._fit_setup(X, y)

        self._fit(X, y)

        # this should happen last
        self.is_fitted = True
        return self

    @final
    def predict(self, X) -> np.ndarray:
        """Predicts target variable for time series in X.

        Parameters
        ----------
        X : np.ndarray or list
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``
            other types are allowed and converted into one of the above.

            Different estimators have different capabilities to handle different
            types of input. If ``self.get_tag("capability:multivariate")`` is False,
            they cannot handle multivariate series, so either ``n_channels == 1`` is
            true or X is 2D of shape ``(n_cases, n_timepoints)``. If ``self.get_tag(
            "capability:unequal_length")`` is False, they cannot handle unequal
            length input. In both situations, a ``ValueError`` is raised if X has a
            characteristic that the estimator does not have the capability for is
            passed.

        Returns
        -------
        predictions : np.ndarray
            1D np.array of float, of shape (n_cases) - predicted regression labels
            indices correspond to instance indices in X
        """
        self._check_is_fitted()
        X = self._preprocess_collection(X, store_metadata=False)
        self._check_shape(X)
        return self._predict(X)

    @final
    def fit_predict(self, X, y) -> np.ndarray:
        """Fits the regressor and predicts class labels for X.

        fit_predict produces prediction estimates using just the train data.
        By default, this is through 10x cross validation, although some estimators may
        utilise specialist techniques such as out-of-bag estimates or leave-one-out
        cross-validation.

        Regressors which override _fit_predict will have the
        ``capability:train_estimate`` tag set to True.

        Generally, this will not be the same as fitting on the whole train data
        then making train predictions. To do this, you should call fit(X,y).predict(X)

        Parameters
        ----------
        X : np.ndarray or list
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. other types are
            allowed and converted into one of the above.

            Different estimators have different capabilities to handle different
            types of input. If `self.get_tag("capability:multivariate")`` is False,
            they cannot handle multivariate series, so either ``n_channels == 1`` is
            true or X is 2D of shape ``(n_cases, n_timepoints)``. If ``self.get_tag(
            "capability:unequal_length")`` is False, they cannot handle unequal
            length input. In both situations, a ``ValueError`` is raised if X has a
            characteristic that the estimator does not have the capability for is
            passed.
        y : np.ndarray
            1D np.array of float, of shape ``(n_cases)`` - regression targets
            (ground truth) for fitting indices corresponding to instance indices in X.

        Returns
        -------
        predictions : np.ndarray
            1D np.array of float, of shape (n_cases) - predicted regression labels
            indices correspond to instance indices in X
        """
        X, y = self._fit_setup(X, y)

        y_pred = self._fit_predict(X, y)

        # this should happen last
        self.is_fitted = True
        return y_pred

    def score(self, X, y, metric="r2", metric_params=None) -> float:
        """Scores predicted labels against ground truth labels on X.

        Parameters
        ----------
        X : np.ndarray or list
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. other types are
            allowed and converted into one of the above.

            Different estimators have different capabilities to handle different
            types of input. If `self.get_tag("capability:multivariate")`` is False,
            they cannot handle multivariate series, so either ``n_channels == 1`` is
            true or X is 2D of shape ``(n_cases, n_timepoints)``. If ``self.get_tag(
            "capability:unequal_length")`` is False, they cannot handle unequal
            length input. In both situations, a ``ValueError`` is raised if X has a
            characteristic that the estimator does not have the capability for is
            passed.
        y : np.ndarray
            1D np.array of float, of shape ``(n_cases)`` - regression targets
            (ground truth) for fitting indices corresponding to instance indices in X.
        metric : Union[str, callable], default="r2",
            Defines the scoring metric to test the fit of the model. For supported
            strings arguments, check `sklearn.metrics.get_scorer_names`.
        metric_params : dict, default=None,
            Contains parameters to be passed to the scoring function. If None, no
            parameters are passed.

        Returns
        -------
        score : float
            MSE score of predict(X) vs y
        """
        self._check_is_fitted()
        y = self._check_y(y, len(X))
        _metric_params = metric_params
        if metric_params is None:
            _metric_params = {}
        if isinstance(metric, str):
            __names = get_scorer_names()
            if metric not in __names:
                raise ValueError(
                    f"Metric {metric} is incompatible with `sklearn.metrics.get_scorer`"
                    "function. Valid list of metrics can be obtained using "
                    "the `sklearn.metrics.get_scorer_names` function."
                )
            scorer = get_scorer(metric)
            return scorer._score_func(y, self.predict(X), **_metric_params)
        elif callable(metric):
            return metric(y, self.predict(X), **_metric_params)
        else:
            raise ValueError(
                "The metric parameter should be either a string or a callable"
                f", but got {metric} of type {type(metric)}"
            )

    @abstractmethod
    def _fit(self, X, y):
        """Fit time series regressor to training data.

        Abstract method, must be implemented.

        Parameters
        ----------
        X : Train data
            guaranteed to be of a type in self.get_tag("X_inner_type")
            if ``self.get_tag("X_inner_type")`` equals "numpy3D":
                3D np.ndarray of shape ``(n_cases, n_channels, n_timepoints)``
            if ``self.get_tag("X_inner_type")`` equals "np-list":
                list of 2D np.ndarray of shape ``(n_cases)``
        y : np.ndarray
            1D np.array of float, of shape ``(n_cases)`` - regression targets for
            fitting indices corresponding to instance indices in X.

        Returns
        -------
        self : BaseRegressor
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes ending in "_"
        """
        ...

    @abstractmethod
    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Abstract method, must be implemented.

        Parameters
        ----------
        X : Train data
            guaranteed to be of a type in self.get_tag("X_inner_type")
            if ``self.get_tag("X_inner_type")`` equals "numpy3D":
                3D np.ndarray of shape ``(n_cases, n_channels, n_timepoints)``
            if ``self.get_tag("X_inner_type")`` equals "np-list":
                list of 2D np.ndarray of shape ``(n_cases)``

        Returns
        -------
        predictions : np.ndarray
            1D np.array of float, of shape (n_cases) - predicted regression labels
            indices correspond to instance indices in X
        """
        ...

    def _fit_predict(self, X, y) -> np.ndarray:
        """Fits and predicts labels for sequences in X.

        Parameters
        ----------
        X : Train data
            guaranteed to be of a type in self.get_tag("X_inner_type")
            if ``self.get_tag("X_inner_type")`` equals "numpy3D":
                3D np.ndarray of shape ``(n_cases, n_channels, n_timepoints)``
            if ``self.get_tag("X_inner_type")`` equals "np-list":
                list of 2D np.ndarray of shape ``(n_cases)``
        y : np.ndarray
            1D np.array of float, of shape ``(n_cases)`` - regression targets
            (ground truth) for fitting indices corresponding to instance indices in X.

        Returns
        -------
        predictions : np.ndarray
            1D np.array of float, of shape (n_cases) - predicted regression labels
            indices correspond to instance indices in X
        """
        # fit the regressor
        self._fit(X, y)

        # predict using cross-validation
        random_state = getattr(self, "random_state", None)
        estimator = _clone_estimator(self, random_state)

        return cross_val_predict(
            estimator,
            X=X,
            y=y,
            cv=10,
            method="predict",
            n_jobs=self._n_jobs,
        )

    def _fit_setup(self, X, y):
        # reset estimator at the start of fit
        self.reset()

        X = self._preprocess_collection(X)
        y = self._check_y(y, self.metadata_["n_cases"])

        # return processed X and y
        return X, y

    def _check_y(self, y, n_cases):
        # Check y valid input for regression
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError(
                f"y must be a np.array or a pd.Series, but found type: {type(y)}"
            )
        if isinstance(y, np.ndarray) and y.ndim > 1:
            raise TypeError(f"y must be 1-dimensional, found {y.ndim} dimensions")

        # Check matching number of labels
        n_labels = len(y)
        if n_cases != n_labels:
            raise ValueError(
                f"Mismatch in number of cases. Found X = {n_cases} and y = {n_labels}"
            )

        y_type = type_of_target(y)
        if y_type != "continuous" and y_type != "binary" and y_type != "multiclass":
            raise ValueError(
                f"y type is {y_type} which is not valid for regression. "
                f"Should be continuous, binary or multiclass according to "
                f"sklearn.utils.multiclass.type_of_target"
            )

        if isinstance(y, pd.Series):
            y = pd.Series.to_numpy(y)

        if any([isinstance(label, str) for label in y]):
            raise ValueError(
                "y contains strings, cannot fit a regressor. If suitable, convert "
                "to floats or consider classification."
            )

        return y.astype(float)
