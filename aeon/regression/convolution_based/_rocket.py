"""RandOm Convolutional KErnel Transform (Rocket) regressor.

Pipeline regressor using the Rocket transformer and RidgeCV estimator.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["RocketRegressor"]

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeon.base._base import _clone_estimator
from aeon.regression.base import BaseRegressor
from aeon.transformations.collection.convolution_based import Rocket


class RocketRegressor(BaseRegressor):
    """
    Rocket transformer using RidgeCV regressor.

    This regressor transforms the input data using the Rocket [1]_ transformer
    extracting features from randomly generated kernels, performs a Standard scaling
    and fits a sklearn regressor using the transformed data (default regressor is
    RidgeCV).

    Parameters
    ----------
    n_kernels : int, default=10,000
        The number of kernels for the Rocket transform.
    estimator : sklearn compatible regressor or None, default=None
        The estimator used. If None, a RidgeCV(alphas=np.logspace(-3, 3, 10)) is used.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.

    See Also
    --------
    Rocket
        Rocket transformers are in transformations/collection.
    RocketClassifier

    References
    ----------
    .. [1] Dempster, A., Petitjean, F. and Webb, G.I., 2020. ROCKET: exceptionally fast
        and accurate time series classification using random convolutional kernels.
        Data Mining and Knowledge Discovery, 34(5), pp.1454-1495.

    Examples
    --------
    >>> from aeon.regression.convolution_based import RocketRegressor
    >>> from aeon.datasets import load_covid_3month
    >>> X_train, y_train = load_covid_3month(split="train")
    >>> X_test, y_test = load_covid_3month(split="test")
    >>> reg = RocketRegressor(n_kernels=500)
    >>> reg.fit(X_train, y_train)
    RocketRegressor(n_kernels=500)
    >>> y_pred = reg.predict(X_test)
    """

    _tags = {
        "capability:multithreading": True,
        "algorithm_type": "convolution",
        "X_inner_type": "numpy3D",
    }

    def __init__(
        self,
        n_kernels=10000,
        estimator=None,
        random_state=None,
        n_jobs=1,
    ):
        self.n_kernels = n_kernels
        self.random_state = random_state
        self.estimator = estimator
        self.n_jobs = n_jobs

        super().__init__()

    def _fit(self, X, y):
        """Fit Rocket variant to training data.

        Parameters
        ----------
        X : 3D np.ndarray
            The training data of shape = (n_cases, n_channels, n_timepoints).
        y : 3D np.ndarray
            The target variable values, shape = (n_cases,).

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        self._transformer = Rocket(
            n_kernels=self.n_kernels,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._scaler = StandardScaler(with_mean=False)
        self._estimator = _clone_estimator(
            (
                RidgeCV(alphas=np.logspace(-3, 3, 10))
                if self.estimator is None
                else self.estimator
            ),
            self.random_state,
        )

        self.pipeline_ = make_pipeline(
            self._transformer,
            self._scaler,
            self._estimator,
        )
        self.pipeline_.fit(X, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = (n_cases,)
            Predicted class labels.
        """
        return self.pipeline_.predict(X)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        dict or list of dict
            Parameters to create testing instances of the class.
        """
        return {"n_kernels": 20}
