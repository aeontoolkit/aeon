"""MERLIN anomaly detector."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["MERLIN"]

import warnings

import numpy as np
from numba import njit

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.distances import squared_distance
from aeon.utils.numba.general import AEON_NUMBA_STD_THRESHOLD
from aeon.utils.numba.stats import mean, std


class MERLIN(BaseAnomalyDetector):
    """MERLIN anomaly detector.

    MERLIN is a discord discovery algorithm that uses a sliding window to find the
    most anomalous subsequence in a time series [1]_. The algorithm is based on the
    Euclidean distance between subsequences of the time series.

    Parameters
    ----------
    min_length : int, default=5
        Minimum length of the subsequence to search for. Must be at least 4.
    max_length : int, default=50
        Maximum length of the subsequence to search for. Must be at half the length
        of the time series or less.
    max_iterations : int, default=500
        Maximum number of DRAG iterations to find an anomalous sequence for each
        length. If no anomaly is found, the algorithm will move to the next length
        and reset ``r``.

    References
    ----------
    .. [1] Nakamura, M. Imamura, R. Mercer and E. Keogh, "MERLIN: Parameter-Free
           Discovery of Arbitrary Length Anomalies in Massive Time Series
           Archives," 2020 IEEE International Conference on Data Mining (ICDM),
           Sorrento, Italy, 2020, pp. 1190-1195.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.anomaly_detection import MERLIN
    >>> X = np.array([1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 5, 1, 2, 3, 4])
    >>> detector = MERLIN(min_length=4, max_length=5)
    >>> detector.fit_predict(X)
    array([False, False, False, False,  True,  True, False, False, False,
           False, False, False, False, False, False, False])
    """

    def __init__(self, min_length=5, max_length=50, max_iterations=500):
        self.min_length = min_length
        self.max_length = max_length
        self.max_iterations = max_iterations

        super().__init__(axis=1)

    def _predict(self, X):
        X = X.squeeze()

        if X.shape[0] < self.min_length:
            raise ValueError(
                f"Series length of X {X.shape[0]} is less than min_length "
                f"{self.min_length}"
            )
        elif self.min_length > self.max_length:
            raise ValueError(
                f"min_length {self.min_length} must be less than max_length "
                f"{self.max_length}"
            )
        elif self.min_length < 4:
            raise ValueError("min_length must be at least 4")
        elif int(len(X) / 2) < self.max_length:
            raise ValueError(
                f"Series length of X {X.shape[0]} must be at least double max_length "
                f"{self.max_length}"
            )

        for i in range(X.shape[0] - self.min_length + 1):
            if std(X[i : i + self.min_length]) > AEON_NUMBA_STD_THRESHOLD:
                warnings.warn(
                    "There is region close to constant that will cause the results "
                    "to be unstable. It is suggested to delete the constant region "
                    "or try again with a longer min_length.",
                    stacklevel=2,
                )

        lengths = np.linspace(
            self.min_length,
            self.max_length,
            num=self.max_length - self.min_length + 1,
            dtype=np.int32,
        )

        r = 2 * np.sqrt(self.min_length)
        distances = np.full(len(lengths), -1.0)
        indicies = np.full(len(lengths), -1)

        indicies[0], distances[0] = self._find_index(X, lengths[0], r, np.multiply, 0.5)

        for i in range(1, min(5, len(lengths))):
            r = distances[i - 1] * 0.99
            indicies[i], distances[i] = self._find_index(
                X, lengths[i], r, np.multiply, 0.99
            )

        for i in range(min(5, len(lengths)), len(lengths)):
            m = mean(distances[i - 5 : i])
            s = std(distances[i - 5 : i])
            r = m - 2 * s
            indicies[i], distances[i] = self._find_index(
                X, lengths[i], r, np.subtract, s
            )

        anomalies = np.zeros(X.shape[0], dtype=bool)
        for i in indicies:
            if i > -1:
                anomalies[i] = True

        return anomalies

    def _find_index(self, X, length, r, mod_func, mod_val):
        it = 0
        distance = -1.0
        index = -1

        while distance < 0:
            # If the algorithm is taking too long, move to the next length and reset r
            if it > self.max_iterations:
                return -1, 2 * np.sqrt(length)

            index, distance = self._drag(X, length, r)
            r = mod_func(r, mod_val)
            it += 1

        return index, distance

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _drag(X, length, discord_range):
        C = []
        data = np.zeros((X.shape[0] - length + 1, length))
        for i in range(X.shape[0] - length + 1):
            is_candidate = True
            data[i] = X[i : i + length]
            sstd = std(data[i]) + AEON_NUMBA_STD_THRESHOLD
            data[i] = (data[i] - mean(data[i])) / sstd

            for n, j in enumerate(C):
                if (
                    np.abs(i - j) >= length
                    and squared_distance(data[i], data[j]) < discord_range
                ):
                    del C[n]
                    is_candidate = False
                    break

            if is_candidate:
                C.append(i)

        if len(C) == 0:
            return -1, -1

        D = [np.inf] * len(C)
        del_list = [False] * len(C)
        for i in range(X.shape[0] - length + 1):
            for n, j in enumerate(C):
                if np.abs(i - j) >= length:
                    d = squared_distance(data[i], data[j])
                    if d < discord_range:
                        del_list[n] = True
                    else:
                        D[n] = np.minimum(D[n], d)

            for n, j in enumerate(del_list):
                if j:
                    del C[n]
                    del D[n]
                    del del_list[n]

        if len(C) == 0:
            return -1, -1

        all_inf = True
        for n in range(len(C)):
            if D[n] == np.inf:
                D[n] = -1
            else:
                all_inf = False

        if all_inf:
            return -1, -1

        d_max = int(np.argmax(np.array(D)))
        return C[d_max], np.sqrt(D[d_max])

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
        return {"min_length": 4, "max_length": 7}
