"""COPOD for anomaly detection."""

__maintainer__ = []
__all__ = ["COPOD"]

from typing import Union

import numpy as np

from aeon.anomaly_detection._pyodadapter import PyODAdapter
from aeon.utils.validation._dependencies import _check_soft_dependencies


class COPOD(PyODAdapter):
    """COPOD for anomaly detection.

    This class implements the COPOD using PyODAdadpter to be used in the aeon framework.
    The parameter `n_jobs` is passed to COPOD model from PyOD, `window_size` and
    `stride` are used to construct the sliding windows.

    Parameters
    ----------
    n_jobs : int, default=1
        The number of jobs to run in parallel for the COPOD model.

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

    def __init__(self, n_jobs: int = 1, window_size: int = 10, stride: int = 1):
        _check_soft_dependencies(*self._tags["python_dependencies"])
        from pyod.models.copod import COPOD

        model = COPOD(n_jobs=n_jobs)
        self.n_jobs = n_jobs
        super().__init__(model, window_size=window_size, stride=stride)

    def _fit(self, X: np.ndarray, y: Union[np.ndarray, None] = None) -> None:
        super()._fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return super()._predict(X)

    def _fit_predict(
        self, X: np.ndarray, y: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        return super()._fit_predict(X, y)

    @classmethod
    def _get_test_params(cls, parameter_set="default") -> dict:
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return {}
