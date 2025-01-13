"""OneClassSVM anomaly detector."""

__all__ = ["OneClassSVM"]

from typing import Optional

import numpy as np
from sklearn.svm import OneClassSVM as OCSVM

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.utils.windowing import reverse_windowing, sliding_windows


class OneClassSVM(BaseAnomalyDetector):
    """OneClassSVM for anomaly detection.

    This class implements the OneClassSVM algorithm for anomaly detection
    from sklearn to be used in the aeon framework. All parameters are passed to
    the sklearn ``OneClassSVM`` except for `window_size` and `stride`, which are used to
    construct the sliding windows.

    The documentation for parameters has been adapted from
    (https://scikit-learn.org/dev/modules/generated/sklearn.svm.OneClassSVM.html).
    Here, `X` refers to the set of sliding windows extracted from the time series
    using :func:`aeon.utils.windowing.sliding_windows` with the parameters
    ``window_size`` and ``stride``. The internal `X` has the shape
    `(n_windows, window_size * n_channels)`.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
        default='rbf'
         Specifies the kernel type to be used in the algorithm.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.

    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Must be non-negative. Ignored by all other kernels.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features
        - if float, must be non-negative.

        .. versionchanged:: 0.22
           The default value of ``gamma`` changed from 'auto' to 'scale'.

    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    nu : float, default=0.5
        An upper bound on the fraction of training
        errors and a lower bound of the fraction of support
        vectors. Should be in the interval (0, 1]. By default 0.5
        will be taken.

    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
        See https://scikit-learn.org/dev/modules/svm.html#shrinking-svm.

    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).

    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

    window_size : int, default=10
        Size of the sliding window.

    stride : int, default=1
        Stride of the sliding window.

    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:missing_values": False,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        nu=0.5,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=0.001,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1,
        window_size: int = 10,
        stride: int = 1,
    ):
        super().__init__(axis=0)
        self.nu = nu
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.window_size = window_size
        self.stride = stride

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "OneClassSVM":
        self._check_params(X)

        _X, _ = sliding_windows(
            X, window_size=self.window_size, stride=self.stride, axis=0
        )
        self._inner_fit(_X)

        return self

    def _check_params(self, X: np.ndarray) -> None:
        if self.window_size < 1 or self.window_size > X.shape[0]:
            raise ValueError(
                "The window size must be at least 1 and at most the length of the "
                "time series."
            )

        if self.stride < 1 or self.stride > self.window_size:
            raise ValueError(
                "The stride must be at least 1 and at most the window size."
            )

    def _inner_fit(self, X: np.ndarray) -> None:
        self.estimator_ = OCSVM(
            nu=self.nu,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            tol=self.tol,
            shrinking=self.shrinking,
            cache_size=self.cache_size,
            verbose=self.verbose,
            max_iter=self.max_iter,
        )
        self.estimator_.fit(X)

    def _predict(self, X) -> np.ndarray:
        _X, padding = sliding_windows(
            X, window_size=self.window_size, stride=self.stride, axis=0
        )

        point_anomaly_scores = self._inner_predict(_X, padding)

        return point_anomaly_scores

    def _fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self._check_params(X)
        _X, padding = sliding_windows(
            X, window_size=self.window_size, stride=self.stride, axis=0
        )
        self._inner_fit(_X)
        point_anomaly_scores = self._inner_predict(_X, padding)
        return point_anomaly_scores

    def _inner_predict(self, X: np.ndarray, padding: int) -> np.ndarray:
        anomaly_scores = self.estimator_.score_samples(X)

        point_anomaly_scores = reverse_windowing(
            anomaly_scores, self.window_size, np.nanmean, self.stride, padding
        )

        point_anomaly_scores = (point_anomaly_scores - point_anomaly_scores.min()) / (
            point_anomaly_scores.max() - point_anomaly_scores.min()
        )

        return point_anomaly_scores
