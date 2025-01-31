__maintainer__ = []

import numpy as np
from numba import njit

from aeon.distances._utils import reshape_pairwise_to_multiple


@njit(cache=True, fastmath=True)
def swale_distance(
    x: np.ndarray,
    y: np.ndarray,
    gapc: float = 1.0,
    rewardm: float = 1.0,
    epsilon: float = 1.0,
) -> float:
    """
    Calculate the Swale distance between two time series.

    Parameters
    ----------
    x : numpy.ndarray
        First time series.
    y : numpy.ndarray
        Second time series.
    gapc : float
        Gap cost.
    rewardm : float
        Reward for match.
    epsilon : float
        Epsilon value.

    Returns
    -------
    float
        The Swale distance between the time series.

    Raises
    ------
    ValueError
        If x and y have different dimensions.

    Examples
    --------
    >>> time_series1 = np.array([1, 2, 3, 4, 5, 6, 7])
    >>> time_series2 = np.array([1, 2, 3, 4])
    >>> print(swale_distance(time_series1, time_series2, 1, 2, 1))
    11
    """
    if x.ndim == 1 and y.ndim == 1:
        return _univariate_swale_distance(x, y, gapc, rewardm, epsilon)
    if x.ndim == 2 and y.ndim == 2:
        return _swale_distance(x, y, gapc, rewardm, epsilon)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _swale_distance(
    x: np.ndarray,
    y: np.ndarray,
    gapc: float = 1.0,
    rewardm: float = 1.0,
    epsilon: float = 1.0,
) -> float:
    n_series1 = x.shape[0]
    n_series2 = y.shape[0]

    distance = 0.0
    min_val = min(n_series1, n_series2)
    for i in range(min_val):
        distance += _univariate_swale_distance(x[i], y[i], gapc, rewardm, epsilon)

    for j in range(min_val, max(n_series1, n_series2)):
        if n_series1 > n_series2:
            distance += _univariate_swale_distance(
                x[j], np.array((), dtype=np.float64), gapc, rewardm, epsilon
            )
        else:
            distance += _univariate_swale_distance(
                np.array((), dtype=np.float64), y[j], gapc, rewardm, epsilon
            )

    return float(distance)


@njit(cache=True, fastmath=True)
def _univariate_swale_distance(
    x: np.ndarray,
    y: np.ndarray,
    gapc: float = 1.0,
    rewardm: float = 1.0,
    epsilon: float = 1.0,
) -> float:
    m = float(len(x))
    n = float(len(y))

    if m == 0:
        return n * gapc
    elif n == 0:
        return m * gapc
    elif abs(x[0] - y[0]) <= epsilon:
        return rewardm + _univariate_swale_distance(
            x[1:], y[1:], gapc=gapc, rewardm=rewardm, epsilon=epsilon
        )
    else:
        option1 = gapc + _univariate_swale_distance(
            x[1:], y, gapc=gapc, rewardm=rewardm, epsilon=epsilon
        )
        option2 = gapc + _univariate_swale_distance(
            x, y[1:], gapc=gapc, rewardm=rewardm, epsilon=epsilon
        )
        return max(option1, option2)


@njit(cache=True, fastmath=True)
def _swale_pairwise_distance(
    x: np.ndarray, gapc: float = 1.0, rewardm: float = 1.0, epsilon: float = 1.0
) -> np.ndarray:
    n_cases = x.shape[0]
    distances = np.zeros((n_cases, n_cases))

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            distances[i, j] = swale_distance(
                x[i], x[j], gapc=gapc, rewardm=rewardm, epsilon=epsilon
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def swale_pairwise_distance(
    x: np.ndarray,
    y: np.ndarray,
    gapc: float = 1.0,
    rewardm: float = 1.0,
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Calculate pairwise Swale distances between two sets of time series.

    Parameters
    ----------
    x : numpy.ndarray
        First set of time series.
    y : numpy.ndarray
        Second set of time series.
    gapc : float
        Gap cost.
    rewardm : float
        Reward for match.
    epsilon : float
        Epsilon value.

    Returns
    -------
    numpy.ndarray
        Pairwise Swale distances between the two sets of time series.

    Examples
    --------
    >>> X = np.array([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]])
    >>> print(swale_pairwise_distance(X, y_univariate, 1, 2, 1))
    [[12.]
     [12.]
     [12.]]
    >>> print(swale_pairwise_distance(X, None, 1, 2, 1))
    [[0. 6. 6.]
     [6. 0. 6.]
     [6. 6. 0.]]
    """
    if y is None:
        if x.ndim == 3:
            return _swale_pairwise_distance(x, gapc, rewardm, epsilon)
        elif x.ndim == 2:
            _x = x.reshape((x.shape[0], 1, x.shape[1]))
            return _swale_pairwise_distance(_x, gapc, rewardm, epsilon)
        raise ValueError("X must be 2D or 3D array")

    _x, _y = reshape_pairwise_to_multiple(x, y)
    return _swale_from_multiple_to_multiple_distance(_x, _y, gapc, rewardm, epsilon)


@njit(cache=True, fastmath=True)
def _swale_from_multiple_to_multiple_distance(
    x: np.ndarray,
    y: np.ndarray,
    gapc: float = 1.0,
    rewardm: float = 1.0,
    epsilon: float = 1.0,
) -> np.ndarray:
    n_cases = x.shape[0]
    m_cases = y.shape[0]
    distances = np.zeros((n_cases, m_cases))

    for i in range(n_cases):
        for j in range(m_cases):
            distances[i, j] = swale_distance(x[i], y[j], gapc, rewardm, epsilon)

    return distances
