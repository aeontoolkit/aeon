"""Tests for STRAY (Search TRace AnomalY) outlier estimator."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["STRAY"]


import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors

from aeon.anomaly_detection.base import BaseAnomalyDetector


class STRAY(BaseAnomalyDetector):
    """STRAY: robust anomaly detection in data streams with concept drift.

    This is based on STRAY (Search TRace AnomalY) [1]_, which is a modification
    of HDoutliers [2]_. HDoutliers is a powerful algorithm for the detection of
    anomalous observations in a dataset, which has (among other advantages) the
    ability to detect clusters of outliers in multidimensional data without
    requiring a model of the typical behavior of the system. However, it suffers
    from some limitations that affect its accuracy. STRAY is an extension of
    HDoutliers that uses extreme value theory for the anomolous threshold
    calculation, to deal with data streams that exhibit non-stationary behavior.

    Parameters
    ----------
    alpha : float, default=0.01
        Threshold for determining the cutoff for outliers. Observations are
        considered outliers if they fall in the (1 - alpha) tail of
        the distribution of the nearest-neighbor distances between exemplars.
    k : int, default=10
        Number of neighbours considered.
    knn_algorithm : str {"auto", "ball_tree", "kd_tree", "brute"}, optional
        (default="brute")
        Algorithm used to compute the nearest neighbors, from
        sklearn.neighbors.NearestNeighbors
    p : float, default=0.5
        Proportion of possible candidates for outliers. This defines the starting point
        for the bottom up searching algorithm.
    size_threshold : int, default=50
        Sample size to calculate an emperical threshold.
    outlier_tail : str {"min", "max"}, default="max"
        Direction of the outlier tail.

    References
    ----------
    .. [1] Talagala, Priyanga Dilini, Rob J. Hyndman, and Kate Smith-Miles.
           "Anomaly detection in high-dimensional data." Journal of Computational
           and Graphical Statistics 30.2 (2021): 360-374.
    .. [2] Wilkinson, Leland. "Visualizing big data outliers through
           distributed aggregation." IEEE transactions on visualization and
           computer graphics 24.1 (2017): 256-266.

    Examples
    --------
    >>> from aeon.anomaly_detection import STRAY
    >>> from aeon.datasets import load_airline
    >>> import numpy as np
    >>> X = load_airline()
    >>> detector = STRAY(k=3)
    >>> y = detector.fit_predict(X)
    >>> y[:5]
    array([False, False, False, False, False])
    """

    _tags = {
        "capability:multivariate": True,
        "capability:missing_values": True,
        "X_inner_type": "np.ndarray",
    }

    def __init__(
        self,
        alpha: float = 0.01,
        k: int = 10,
        knn_algorithm: str = "brute",
        p: float = 0.5,
        size_threshold: int = 50,
        outlier_tail: str = "max",
    ):
        self.alpha = alpha
        self.k = k
        self.knn_algorithm = knn_algorithm
        self.p = p
        self.size_threshold = size_threshold
        self.outlier_tail = outlier_tail

        super().__init__(axis=0)

    def _predict(self, X, y=None) -> npt.ArrayLike:
        idx_dropna = np.array(
            [i for i in range(X.shape[0]) if not np.isnan(X[i]).any()]
        )
        X_dropna = X[idx_dropna,]

        outliers = self._find_outliers_kNN(X_dropna, X_dropna.shape[0])

        # adjusted back to length r, for missing data
        slice_ = [i in outliers["idx_outliers"] for i in range(X_dropna.shape[0])]
        idx_outliers = idx_dropna[slice_]  # index values from 1:r
        outlier_bool = np.array(
            [1 if i in idx_outliers else 0 for i in range(X.shape[0])]
        )

        return outlier_bool.astype(bool)

    def _find_outliers_kNN(self, X: np.ndarray, n: int) -> dict:
        """Find outliers using kNN distance with maximum gap.

        Parameters
        ----------
        X : np.ndarray
            Data for anomaly detection (time series).
        n : int
            The number of rows remaining in X when NA's are removed.

        Returns
        -------
        dict of index of outliers and the outlier scores
        """
        nbrs = NearestNeighbors(
            n_neighbors=n if self.k >= n else self.k + 1,
            algorithm=self.knn_algorithm,
        ).fit(X)
        distances, _ = nbrs.kneighbors(X)

        if self.k == 1:
            d = distances[:, 1]
        else:
            diff = np.apply_along_axis(np.diff, 1, distances)
            d = distances[range(n), np.apply_along_axis(np.argmax, 1, diff) + 1]

        out_index = self._find_threshold(d, n)
        return {"idx_outliers": out_index, "out_scores": d}

    def _find_threshold(self, outlier_score: npt.ArrayLike, n: int) -> npt.ArrayLike:
        """Find Outlier Threshold.

        Parameters
        ----------
        outlier_score : np.ArrayLike
            The outlier scores determined by k nearest neighbours distance
        n : int
            The number of rows remaining in X when NA's are removed.

        Returns
        -------
        array of indices of the observations determined to be outliers.
        """
        if self.outlier_tail == "min":
            outlier_score = -outlier_score

        order = np.argsort(outlier_score)
        gaps = np.append(0, np.diff(outlier_score[order]))
        n4 = int(max(min(self.size_threshold, np.floor(n / 4)), 2))

        J = np.array(range(2, n4 + 1))
        start = int(max(np.floor(n * (1 - self.p)), 1))

        ghat = [
            0.0 if i < start else sum((J / (n4 - 1)) * gaps[i - J + 1])
            for i in range(n)
        ]

        log_alpha = np.log(1 / self.alpha)
        bound = np.inf

        for i in range(start, n):
            if gaps[i] > log_alpha * ghat[i]:
                bound = outlier_score[order][i - 1]
                break

        return np.where(outlier_score > bound)[0]
