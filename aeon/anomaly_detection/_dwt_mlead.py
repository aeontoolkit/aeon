"""DWT-MLEAD anomaly detector."""

__maintainer__ = ["SebastianSchmidl"]
__all__ = ["DWT_MLEAD"]

import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.covariance import EmpiricalCovariance

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.utils.numba.wavelets import multilevel_haar_transform


def _pad_series(x: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Pad input signal to the next power of 2 using periodic padding mode."""
    n = x.shape[0]
    exp = np.ceil(np.log2(n))
    m = int(np.power(2, exp))
    return np.pad(x, (0, m - n), mode="wrap"), n, m


def _combine_alternating(xs: list[Any], ys: list[Any]) -> Iterable[Any]:
    """Combine two lists by alternating their elements."""
    for x, y in zip(xs, ys):
        yield x
        yield y


class DWT_MLEAD(BaseAnomalyDetector):
    """DWT-MLEAD anomaly detector.

    DWT-MLEAD is an anomaly detection algorithm that uses the Discrete Wavelet Transform
    (DWT) and Maximum Likelihood Estimation (MLE) to detect anomalies in univariate
    time series. The algorithm performs mutli-level DWT using the Haar wavelet, slides
    windows over the DWT coefficients, and estimates the likelihood of each window
    using a Gaussian distribution. Anomalies are detected by comparing the likelihoods
    to a quantile boundary in each level and passing down the anomaly counts to the
    individual time points, which we use as anomaly scores. The original paper [1]_
    subsequently clusters the anomalies to determine the anomaly centers. This step is
    not implemented in this version.

    Parameters
    ----------
    start_level : int, default=3
        The level at which to start the anomaly detection. Must be >= 0 and less than
        log_2(n_timepoints).
    quantile_boundary_type : str, default='percentile'
        The type of boundary to use for the quantile. Must be 'percentile',
        'monte-carlo' is not implemented yet.
    quantile_epsilon : float, default=0.01
        The epsilon value for the quantile boundary. Must be in [0, 1].

    Notes
    -----
    This implementation does not exactly match the original paper [1]_. We make the
    following changes:

    - We use window sizes for the DWT coefficients that decrease with the level number
      because otherwise we would have too few items to slide the window over.
    - We exclude the highest level coefficients because they contain only a single entry
      and are, thus, not suitable for sliding a window of length 2 over it.
    - We have not implemented the Monte Carlo quantile boundary type yet.
    - We do not perform the anomaly clustering step to determine the anomaly centers.
      Instead, we return the anomaly scores for each timestep in the original time
      series.

    References
    ----------
    .. [1] Thill, Markus, Wolfgang Konen, and Thomas Bäck. "Time Series Anomaly
           Detection with Discrete Wavelet Transforms and Maximum Likelihood
           Estimation." In Proceedings of the International Conference on Time Series
           (ITISE). Granada, Spain, 2017.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.anomaly_detection import DWT_MLEAD
    >>> X = np.array([1, 2, 3, 4, 1, 2, 3, 3, 2, 8, 9, 8, 1, 2, 3, 4], dtype=np.float64)
    >>> detector = DWT_MLEAD(
    ...    start_level=1, quantile_boundary_type='percentile', quantile_epsilon=0.01
    ... )
    >>> detector.fit_predict(X)
    array([0., 0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2., 0., 0., 0., 0.])
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:missing_values": False,
    }

    def __init__(
        self,
        start_level: int = 3,
        quantile_boundary_type: str = "percentile",
        quantile_epsilon: float = 0.01,
    ):
        self.start_level = start_level
        self.quantile_boundary_type = quantile_boundary_type
        self.quantile_epsilon = quantile_epsilon

        super().__init__(axis=0)

    def _predict(self, X) -> np.ndarray:
        X = X.squeeze()

        if self.start_level < 0:
            raise ValueError("start_level must be >= 0")
        if self.quantile_boundary_type != "percentile":
            if self.quantile_boundary_type not in ["percentile", "monte-carlo"]:
                raise ValueError(
                    "quantile_boundary_type must be 'percentile' or 'monte-carlo', "
                    f"but is {self.quantile_boundary_type}"
                )
            else:
                raise NotImplementedError(
                    f"The quantile boundary type '{self.quantile_boundary_type}' "
                    "is not implemented yet!"
                )
        if self.quantile_epsilon < 0 or self.quantile_epsilon > 1:
            raise ValueError("quantile_epsilon must be in [0, 1]")

        X, n, m = _pad_series(X)
        max_level = int(np.log2(m))

        if self.start_level >= max_level:
            raise ValueError(
                f"start_level ({self.start_level}) must be less than "
                f"log_2(n_timepoints) ({max_level})"
            )

        # perform multilevel DWT and capture coefficients
        levels, approx_coeffs, detail_coeffs = self._multilevel_dwt(X, max_level)

        # extract anomalies in each level
        window_sizes = np.array(
            [
                max(2, max_level - level - self.start_level + 1)
                for level in range(max_level)
            ],
            dtype=np.int_,
        )
        coef_anomaly_counts = []
        for x, level in zip(
            _combine_alternating(detail_coeffs, approx_coeffs), levels.repeat(2, axis=0)
        ):
            w = window_sizes[level]
            windows = sliding_window_view(x, w)

            p = self._estimate_gaussian_likelihoods(windows)
            a = self._mark_anomalous_windows(p)
            xa = self._reverse_windowing(a, window_length=w, full_length=x.shape[0])
            coef_anomaly_counts.append(xa)

        # aggregate anomaly counts (leaf counters)
        point_anomaly_scores = self._push_anomaly_counts_down_to_points(
            coef_anomaly_counts, m, n
        )
        return point_anomaly_scores

    def _multilevel_dwt(
        self, X: np.ndarray, max_level: int
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        ls_ = np.arange(self.start_level - 1, max_level - 1, dtype=np.int_) + 1
        as_, ds_ = multilevel_haar_transform(X, max_level - 1)
        as_ = as_[self.start_level :]
        ds_ = ds_[self.start_level - 1 :]
        return ls_, as_, ds_

    @staticmethod
    def _estimate_gaussian_likelihoods(x_windows: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=UserWarning)

            # fit gaussion distribution with mean and covariance
            estimator = EmpiricalCovariance(assume_centered=False)
            estimator.fit(x_windows)

            # compute log likelihood for each window x in x_view
            n_windows = x_windows.shape[0]
            p = np.empty(shape=n_windows)
            for i in range(n_windows):
                p[i] = estimator.score(x_windows[i].reshape(1, -1))
        return p

    def _mark_anomalous_windows(self, p: np.ndarray) -> np.ndarray:
        if self.quantile_boundary_type == "percentile":
            z_eps = np.percentile(p, self.quantile_epsilon * 100)
        else:  # self.quantile_boundary_type == "monte-carlo"
            raise ValueError(
                f"The quantile boundary type '{self.quantile_boundary_type}' "
                "is not implemented yet!"
            )

        return p < z_eps

    @staticmethod
    def _reverse_windowing(
        x: np.ndarray, window_length: int, full_length: int
    ) -> np.ndarray:
        mapped = np.full(shape=(full_length, window_length), fill_value=0)
        mapped[: x.shape[0], 0] = x

        for w in range(1, window_length):
            mapped[:, w] = np.roll(mapped[:, 0], w)

        return np.sum(mapped, axis=1)

    @staticmethod
    def _push_anomaly_counts_down_to_points(
        coef_anomaly_counts: list[np.ndarray], m: int, n: int
    ) -> np.ndarray:
        # sum up counters of detail coeffs (orig. D^l) and approx coeffs (orig. C^l)
        anomaly_counts = coef_anomaly_counts[0::2] + coef_anomaly_counts[1::2]

        # extrapolate anomaly counts to the original series' points
        counter = np.zeros(m)
        for ac in anomaly_counts:
            counter += ac.repeat(m // ac.shape[0], axis=0)
        # set event counters with count < 2 to 0
        counter[counter < 2] = 0
        return counter[:n]

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Only supports 'default'-parameter set.

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
            "start_level": 2,
            "quantile_boundary_type": "percentile",
            "quantile_epsilon": 0.01,
        }
