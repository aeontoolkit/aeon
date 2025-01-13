"""k-Means anomaly detector."""

__maintainer__ = ["SebastianSchmidl"]
__all__ = ["KMeansAD"]

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.utils.windowing import reverse_windowing, sliding_windows


class KMeansAD(BaseAnomalyDetector):
    """KMeans anomaly detector.

    The k-Means anomaly detector uses k-Means clustering to detect anomalies in time
    series. The time series is split into windows of a fixed size, and the k-Means
    algorithm is used to cluster these windows. The anomaly score for each time point is
    the average Euclidean distance between the time point's windows and the windows'
    corresponding cluster centers.

    ``k-MeansAD`` supports univariate and multivariate time series. It can also be
    fitted on a clean reference time series and used to detect anomalies in a different
    target time series with the same number of dimensions.

    Parameters
    ----------
    n_clusters : int, default=20
        The number of clusters to use in the k-Means algorithm. The bigger the number
        of clusters, the less noisy the anomaly scores get. However, the number of
        clusters should not be too high, as this can lead to overfitting.

    window_size : int, default=20
        The size of the sliding window used to split the time series into windows. The
        bigger the window size, the bigger the anomaly context is. If it is too big,
        however, the detector marks points anomalous that are not. If it is too small,
        the detector might not detect larger anomalies or contextual anomalies at all.
        If ``window_size`` is smaller than the anomaly, the detector might detect only
        the transitions between normal data and the anomalous subsequence.

    stride : int, default=1
        The stride of the sliding window. The stride determines how many time points
        the windows are spaced appart. A stride of 1 means that the window is moved one
        time point forward compared to the previous window. The larger the stride, the
        fewer windows are created, which leads to noisier anomaly scores.

    random_state : int, default=None
        The random state to use in the k-Means algorithm.

    Notes
    -----
    This implementation is inspired by [1]_. However, the original paper proposes a
    different kind of preprocessing and also uses advanced techniques to post-process
    the clustering.

    References
    ----------
    .. [1] Yairi, Takehisa, Yoshikiyo Kato, and Koichi Hori. "Fault Detection by Mining
           Association Rules from House-Keeping Data." In Proceedings of the
           International Symposium on Artificial Intelligence, Robotics and Automation
           in Space (-SAIRAS), Vol. 6., 2001.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.anomaly_detection import KMeansAD
    >>> X = np.array([1, 2, 3, 4, 1, 2, 3, 3, 2, 8, 9, 8, 1, 2, 3, 4], dtype=np.float64)
    >>> detector = KMeansAD(n_clusters=3, window_size=4, stride=1, random_state=0)
    >>> detector.fit_predict(X)
    array([1.97827709, 2.45374147, 2.51929879, 2.36979677, 2.34826601,
           2.05075554, 2.57611912, 2.87642119, 3.18400743, 3.65060425,
           3.36402514, 3.94053744, 3.65448197, 3.6707922 , 3.70341266,
           1.97827709])

    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:missing_values": False,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        n_clusters: int = 20,
        window_size: int = 20,
        stride: int = 1,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.stride = stride
        self.random_state = random_state

        super().__init__(axis=0)

        self.estimator_: Optional[KMeans] = None

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "KMeansAD":
        self._check_params(X)
        _X, _ = sliding_windows(
            X, window_size=self.window_size, stride=self.stride, axis=0
        )
        self._inner_fit(_X)
        return self

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
        if self.n_clusters < 1:
            raise ValueError("The number of clusters must be at least 1.")

    def _inner_fit(self, X: np.ndarray) -> None:
        self.estimator_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            init="k-means++",
            n_init=10,
            max_iter=300,
            tol=1e-4,
            verbose=0,
            algorithm="lloyd",
        )
        self.estimator_.fit(X)

    def _inner_predict(self, X: np.ndarray, padding: int) -> np.ndarray:
        clusters = self.estimator_.predict(X)
        window_scores = np.linalg.norm(
            X - self.estimator_.cluster_centers_[clusters], axis=1
        )
        point_anomaly_scores = reverse_windowing(
            window_scores, self.window_size, np.nanmean, self.stride, padding
        )
        return point_anomaly_scores

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
        dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {
            "n_clusters": 5,
            "window_size": 10,
            "stride": 1,
            "random_state": 0,
        }
