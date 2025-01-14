"""Implements an adapter for PyOD models to be used in the Aeon framework."""

from __future__ import annotations

__maintainer__ = ["SebastianSchmidl"]
__all__ = ["STOMP"]

import numpy as np

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.utils.windowing import reverse_windowing


class STOMP(BaseAnomalyDetector):
    """STOMP anomaly detector.

    STOMP calculates the matrix profile of a time series which is the distance to the
    nearest neighbor of each subsequence in the time series. The matrix profile is then
    used to calculate the anomaly score for each time point. The larger the distance to
    the nearest neighbor, the more anomalous the time point is.

    STOMP supports univariate time series only.


    Parameters
    ----------
    window_size : int, default=10
        Size of the sliding window.
    ignore_trivial : bool, default=True
        Whether to ignore trivial matches in the matrix profile.
    normalize : bool, default=True
        Whether to normalize the windows before computing the distance.
    p : float, default=2.0
        The p-norm to use for the distance calculation.
    k : int, default=1
        The number of top distances to return.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.anomaly_detection import STOMP  # doctest: +SKIP
    >>> X = np.random.default_rng(42).random((10, 2), dtype=np.float64)
    >>> detector = STOMP(X, window_size=2)  # doctest: +SKIP
    >>> detector.fit_predict(X, axis=0)  # doctest: +SKIP
    array([1.02352234 1.00193038 0.98584441 0.99630753 1.00656619 1.00682081 1.00781515
           0.99709741 0.98878895 0.99723947])

    References
    ----------
    .. [1] Zhu, Yan and Zimmerman, Zachary and Senobari, Nader Shakibay and Yeh,
           Chin-Chia Michael and Funning, Gareth and Mueen, Abdullah and Brisk,
           Philip and Keogh, Eamonn. "Matrix Profile II: Exploiting a Novel
           Algorithm and GPUs to Break the One Hundred Million Barrier for Time
           Series Motifs and Joins." In Proceedings of the 16th International
           Conference on Data Mining (ICDM), 2016.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "fit_is_empty": True,
        "python_dependencies": ["stumpy"],
    }

    def __init__(
        self,
        window_size: int = 10,
        ignore_trivial: bool = True,
        normalize: bool = True,
        p: float = 2.0,
        k: int = 1,
    ):
        self.window_size = window_size
        self.ignore_trivial = ignore_trivial
        self.normalize = normalize
        self.p = p
        self.k = k

        super().__init__(axis=0)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        import stumpy

        self._check_params(X)
        mp = stumpy.stump(
            X[:, 0],
            m=self.window_size,
            ignore_trivial=self.ignore_trivial,
            normalize=self.normalize,
            p=self.p,
            k=self.k,
        )
        point_anomaly_scores = reverse_windowing(mp[:, 0], self.window_size)
        return point_anomaly_scores

    def _check_params(self, X: np.ndarray) -> None:
        if self.window_size < 1 or self.window_size > X.shape[0]:
            raise ValueError(
                "The window size must be at least 1 and at most the length of the "
                "time series."
            )

        if self.k < 1 or self.k > X.shape[0] - self.window_size:
            raise ValueError(
                "The top `k` distances must be at least 1 and at most the length of "
                "the time series minus the window size."
            )

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
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {
            "window_size": 10,
            "ignore_trivial": True,
            "normalize": True,
            "p": 2.0,
            "k": 1,
        }
