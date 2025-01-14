"""Isolation Forest for Anomaly Detection."""

__maintainer__ = []
__all__ = ["IsolationForest"]

from typing import Literal, Optional, Union

import numpy as np

from aeon.anomaly_detection._pyodadapter import PyODAdapter
from aeon.utils.validation._dependencies import _check_soft_dependencies


class IsolationForest(PyODAdapter):
    """Isolation Forest for anomaly detection.

    This class implements the Isolation Forest algorithm for anomaly detection
    using PyODAdadpter to be used in the aeon framework. All parameters are passed to
    the PyOD model ``IForest`` except for `window_size` and `stride`, which are used to
    construct the sliding windows.

    The documentation for parameters has been adapted from the
    [PyOD documentation](https://pyod.readthedocs.io/en/latest/pyod.models.html#id405).
    Here, `X` refers to the set of sliding windows extracted from the time series
    using :func:`aeon.utils.windowing.sliding_windows` with the parameters
    ``window_size`` and ``stride``. The internal `X` has the shape
    `(n_windows, window_size * n_channels)`.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators in the ensemble.

    max_samples : int, float or "auto", default="auto"
        The number of samples to draw from X to train each base estimator.

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.


    random_state : int, np.RandomState or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, default=0
        Controls the verbosity of the tree building process.

    window_size : int, default=10
        Size of the sliding window.

    stride : int, default=1
        Stride of the sliding window.

    """

    _tags = {
        "capability:multivariate": True,
        "capability:univariate": True,
        "capability:missing_values": False,
        "capability:multithreading": True,
        "fit_is_empty": False,
        "python_dependencies": ["pyod"],
    }

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[int, float, Literal["auto"]] = "auto",
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = False,
        n_jobs: int = 1,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: int = 0,
        window_size: int = 10,
        stride: int = 1,
    ):
        _check_soft_dependencies(*self._tags["python_dependencies"])
        from pyod.models.iforest import IForest

        model = IForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        super().__init__(model, window_size, stride)

    def _fit(self, X: np.ndarray, y: Union[np.ndarray, None] = None) -> None:
        super()._fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return super()._predict(X)

    def _fit_predict(
        self, X: np.ndarray, y: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        return super()._fit_predict(X, y)

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
        params : dict
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `IsolationForest(**params)` creates a valid test instance.
        """
        return {
            "n_estimators": 10,
        }
