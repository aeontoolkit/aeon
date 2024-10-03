__maintainer__ = []

import numpy as np
from numba import njit

from aeon.distances._utils import reshape_pairwise_to_multiple


@njit(cache=True, fastmath=True)
def mahalanobis_distance(x: np.ndarray, y: np.ndarray, m: np.ndarray = None) -> float:
    r"""Compute the Mahalanobis distance between two time series.

    The Mahalanobis distance between two time series of length m is defined as:

    .. math::
        md(x, y) = \sum_{i=1}^m (x_i - y_i)^T M (x_i - y_i)

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    m : np.ndarray or None, default=None
        Positive-definite matrix, shape ``(n_channels, n_channels).
        If not provided it is set to inverse of covariance of x.

    Returns
    -------
    float
        Mahalanobis distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
        If x and y do not have same shape.
        If m is not square matrix.
        If dimension of m is not same as dimension of x and y.
        Note that if m is not positive-definite no error wil be reported.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import mahalanobis_distance
    >>> x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])
    >>> mahalanobis_distance(x, y)
    199.99999999999986
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.ndim == 2 and y.ndim == 2:
        _x = x
        _y = y
    elif x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
    else:
        raise ValueError("x and y must be 1D or 2D")
    if m is not None:
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("m must be a square matrix")
        if m.shape[0] != _x.shape[0]:
            raise ValueError("dimension of m must be the same as dimension of x and y")
        _m = m
    else:
        if _x.shape[0] > 1:
            _m = np.linalg.pinv(np.cov(_x))
        else:
            _m = np.linalg.pinv(np.atleast_2d(np.cov(_x[0])))
    return _mahalanobis_distance(_x, _y, _m)


@njit(cache=True, fastmath=True)
def _mahalanobis_distance(x: np.ndarray, y: np.ndarray, m: np.ndarray) -> float:
    distance = 0.0
    length = x.shape[1]
    val = x.shape[0]
    for j in range(length):
        difference = np.zeros(val)
        for i in range(val):
            difference[i] = x[i, j] - y[i, j]
        distance += np.dot(np.dot(difference, m), difference)
    return distance


@njit(cache=True, fastmath=True)
def mahalanobis_pairwise_distance(
    X: np.ndarray, y: np.ndarray = None, m: np.ndarray = None
) -> np.ndarray:
    """Compute the Mahalanobis pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the Mahalanobis pairwise distance between the instances of X
        is calculated.
    m : np.ndarray or None, default=None

    Returns
    -------
    np.ndarray (n_cases, n_cases)
        Mahalanobis pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import mahalanobis_pairwise_distance
    >>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> mahalanobis_pairwise_distance(X)
    array([[ 0. , 16.8, 65.4],
           [16.8,  0. , 16.2],
           [65.4, 16.2,  0. ]])

    >>> import numpy as np
    >>> from aeon.distances import mahalanobis_pairwise_distance
    >>> X = np.array([[[1, 2], [3, 4], [5, 6]], [[2, 3], [4, 5], [6, 7]]])
    >>> mahalanobis_pairwise_distance(X)
    array([[0., 4.],
           [4., 0.]])

    >>> import numpy as np
    >>> from aeon.distances import mahalanobis_pairwise_distance
    >>> X = np.array([[[1, 2], [3, 4], [5, 6]], [[2, 3], [4, 5], [6, 7]]])
    >>> y = np.array([[[2, 3], [3, 4], [4, 5]]])
    >>> mahalanobis_pairwise_distance(X, y)
    >>> array([[0.],
               [4.]])
    """
    if y is None:
        # To self
        if X.ndim == 3:
            return _mahalanobis_pairwise_distance(X, m)
        elif X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _mahalanobis_pairwise_distance(_X, m)
        raise ValueError("X must be 2D or 3D array")
    _x, _y = reshape_pairwise_to_multiple(X, y)
    return _mahalanobis_from_multiple_to_multiple_distance(_x, _y, m)


@njit(cache=True, fastmath=True)
def _mahalanobis_pairwise_distance(X: np.ndarray, m: np.ndarray) -> np.ndarray:
    n_cases = X.shape[0]
    distances = np.zeros((n_cases, n_cases))

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            distances[i, j] = mahalanobis_distance(X[i], X[j], m)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _mahalanobis_from_multiple_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, m: np.ndarray
) -> np.ndarray:
    n_cases = x.shape[0]
    m_cases = y.shape[0]
    distances = np.zeros((n_cases, m_cases))

    for i in range(n_cases):
        for j in range(m_cases):
            distances[i, j] = mahalanobis_distance(x[i], y[j], m)
    return distances
