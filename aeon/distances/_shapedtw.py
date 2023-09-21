# -*- coding: utf-8 -*-
r"""Shape Dynamic time warping (ShapeDTW) between two time series."""
__author__ = ["hadifawaz1999"]

from typing import List, Tuple

import numpy as np
from numba import njit

from aeon.distances import dtw_cost_matrix
from aeon.distances._alignment_paths import compute_min_return_path


@njit(fastmath=True)
def _identity_descriptor(x: np.ndarray) -> np.ndarray:
    """Return the identity function of the given 1-D subsequence.

    Parameters
    ----------
    x : np.ndarray, shape (n_channels, n_timepoints)
    """
    return np.copy(x)


# @njit(fastmath=True)
# def _transform_set_subsequences(
#     X: np.ndarray, descriptor: str = "identity", reach: int = 30
# ) -> np.ndarray:
#     """Decompose all the series into sub-sequences.

#     It applies a transformation over each sub-sequence for each series.

#     Parameters
#     ----------
#     X : np.ndarray
#         A set of multivariate time series of shape =
#         (n_instances, n_channels, n_timepoints)
#     descriptor : str, default=None (if None then identity is used).
#         This defines which transformation is applied on the sub-sequences.
#     reach : int, default=30.
#         This is the length of the sub-sequences.

#     Returns
#     -------
#     out_mts : np.ndarray, shape = (n_instances, new_n_channels, n_timepoints).
#         The output multivariate time series set.
#     """

#     descriptor_map = {"identity": _identity_descriptor}
#     descriptor_function = descriptor_map[descriptor]

#     sliding_window = 2 * reach + 1

#     n_timepoints = X.shape[-1] - 2 * reach
#     n_channels = X.shape[0]

#     dim_desc = descriptor_function(X[0, 0, 0 : 0 + sliding_window]).shape[0]

#     out_mtss = np.zeros(X.shape[0], n_channels * dim_desc, n_timepoints)

#     for i in range(X.shape[0]):
#         out_mtss[i] = _transform_subsequences(
#             x=X[i], descriptor=descriptor, reach=reach
#         )

#     return out_mtss


@njit(fastmath=True)
def _transform_subsequences(
    x: np.ndarray, descriptor: str = "identity", reach: int = 30
) -> np.ndarray:
    """Decompose the series into sub-sequences.

    It applies a transformation over each sub-sequence

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    descriptor : str, default=None (if None then identity is used).
        This defines which transformation is applied on the sub-sequences.
    reach : int, default=30.
        This is the length of the sub-sequences.

    Returns
    -------
    out_mts : np.ndarray, shape = (new_n_channels, n_timepoints+reach*2).
        The output multivariate time series.
    """
    descriptor_map = {"identity": _identity_descriptor}
    descriptor_function = descriptor_map[descriptor]

    # pad the time serie x
    # x = np.pad(x, [[0,0],[reach, reach]], mode="edge")

    sliding_window = reach * 2 + 1
    sliding_window = int(sliding_window)

    # get the output dimension of the subsequence transofrmation s
    dim_desc = descriptor_function(x[0, 0 : 0 + sliding_window]).shape[0]
    # dim_desc = 7

    n_channels = x.shape[0]
    n_timepoints = x.shape[1] - 2 * reach

    # define the output MTS which has the same
    out_mts = np.zeros((n_channels * dim_desc, n_timepoints), dtype=np.float64)

    # loop through each data point
    for i in range(n_timepoints):
        # loop through each dimension of the MTS
        for j in range(n_channels):
            val = descriptor_function(x[j, i : i + sliding_window])
            out_mts[j * dim_desc : (j + 1) * dim_desc, i] = val
    return out_mts


def shape_dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    descriptor: str = "identity",
    reach: int = 30,
) -> float:
    """Compute the ShapeDTW distance function between two series x and y.

    The ShapeDTW distance measure was proposed in [1] and used for time series
    averaging in [2] as well.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
        is used.
    descriptor : str, default=None (if None then identity is used).
        This defines which transformation is applied on the sub-sequences.
    reach : int, default=30.
        This is the length of the sub-sequences.

    Returns
    -------
    float
        DTW distance between x and y, minimum value 0.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    [1] Zhao, Jiaping, and Laurent Itti. "shapedtw: Shape dynamic time warping."
        Pattern Recognition 74 (2018): 171-184.
    [2] Ali Ismail-Fawaz, Hassan Ismail Fawaz, François Petitjean, Maxime Devanne,
        Jonathan Weber, Stefano Berretti, Geoffrey I. Webb and Germain Forestier.
        "ShapeDBA: Generating Effective Time Series Prototypes using ShapeDTW
        Barycenter Averaging" ECML/PKDD Workshop on Advanced Analytics and
        Learning on Temporal Data, Turin, Italy, 2023.
    """
    if x.ndim == 1 and y.ndim == 1:
        x_pad = np.pad(x, [reach, reach], mode="edge")
        y_pad = np.pad(y, [reach, reach], mode="edge")
        _x = x_pad.reshape((1, x_pad.shape[0]))
        _y = y_pad.reshape((1, y_pad.shape[0]))

        return _shape_dtw_distance(
            x=_x, y=_y, window=window, descriptor=descriptor, reach=reach
        )
    if x.ndim == 2 and y.ndim == 2:
        x_pad = np.pad(x, [[0, 0], [reach, reach]], mode="edge")
        y_pad = np.pad(y, [[0, 0], [reach, reach]], mode="edge")

        return _shape_dtw_distance(
            x=x_pad, y=y_pad, window=window, descriptor=descriptor, reach=reach
        )

    raise ValueError("x and y must be 1D or 2D")


@njit(fastmath=True)
def _shape_dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    descriptor: str = "identity",
    reach: int = 30,
) -> float:
    """Compute the ShapeDTW distance function between two series x and y.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
        is used.
    descriptor : str, default=None (if None then identity is used).
        This defines which transformation is applied on the sub-sequences.
    reach : int, default=30.
        This is the length of the sub-sequences.

    Returns
    -------
    float
        DTW distance between x and y, minimum value 0.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
    """
    new_x = _transform_subsequences(x=x, descriptor=descriptor, reach=reach)
    new_y = _transform_subsequences(x=y, descriptor=descriptor, reach=reach)

    shapedtw_cost_mat = dtw_cost_matrix(x=new_x, y=new_y, window=window)

    i = shapedtw_cost_mat.shape[0] - 1
    j = shapedtw_cost_mat.shape[1] - 1

    shapedtw_dist = 0

    while i >= 0 and j >= 0:
        shapedtw_dist += np.square(np.linalg.norm(x[:, reach + i] - y[:, reach + j]))

        a = shapedtw_cost_mat[i - 1, j - 1]
        b = shapedtw_cost_mat[i, j - 1]
        c = shapedtw_cost_mat[i - 1, j]
        if a < b:
            if a < c:
                # a is the minimum
                i -= 1
                j -= 1
            else:
                # c is the minimum
                i -= 1
        else:
            if b < c:
                # b is the minimum
                j -= 1
            else:
                # c is the minimum
                i -= 1

    return np.sqrt(shapedtw_dist)


def shape_dtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    descriptor: str = "identity",
    reach: int = 30,
) -> float:
    """Compute the ShapeDTW cost matrix between two series x and y.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
        is used.
    descriptor : str, default=None (if None then identity is used).
        This defines which transformation is applied on the sub-sequences.
    reach : int, default=30.
        This is the length of the sub-sequences.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        shapedtw cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
    """
    if x.ndim == 1 and y.ndim == 1:
        x_pad = np.pad(x, [reach, reach], mode="edge")
        y_pad = np.pad(y, [reach, reach], mode="edge")
        _x = x_pad.reshape((1, x_pad.shape[0]))
        _y = y_pad.reshape((1, y_pad.shape[0]))

        return _shape_dtw_cost_matrix(
            x=_x, y=_y, window=window, descriptor=descriptor, reach=reach
        )
    if x.ndim == 2 and y.ndim == 2:
        x_pad = np.pad(x, [[0, 0], [reach, reach]], mode="edge")
        y_pad = np.pad(y, [[0, 0], [reach, reach]], mode="edge")

        return _shape_dtw_cost_matrix(
            x=x_pad, y=y_pad, window=window, descriptor=descriptor, reach=reach
        )

    raise ValueError("x and y must be 1D or 2D")


def _shape_dtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    descriptor: str = "identity",
    reach: int = 30,
) -> float:
    """Compute the ShapeDTW cost matrix between two series x and y.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
        is used.
    descriptor : str, default=None (if None then identity is used).
        This defines which transformation is applied on the sub-sequences.
    reach : int, default=30.
        This is the length of the sub-sequences.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        shapedtw cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
    """
    new_x = _transform_subsequences(x=x, descriptor=descriptor, reach=reach)
    new_y = _transform_subsequences(x=y, descriptor=descriptor, reach=reach)

    shapedtw_cost_mat = dtw_cost_matrix(x=new_x, y=new_y, window=window)

    return shapedtw_cost_mat


def shape_dtw_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    descriptor: str = "identity",
    reach: int = 30,
) -> List[Tuple[int, int]]:
    """Compute the ShapeDTW alignment path between two series x and y.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
        is used.
    descriptor : str, default=None (if None then identity is used).
        This defines which transformation is applied on the sub-sequences.
    reach : int, default=30.
        This is the length of the sub-sequences.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
    """
    if x.ndim == 1 and y.ndim == 1:
        x_pad = np.pad(x, [reach, reach], mode="edge")
        y_pad = np.pad(y, [reach, reach], mode="edge")
        _x = x_pad.reshape((1, x_pad.shape[0]))
        _y = y_pad.reshape((1, y_pad.shape[0]))

        return _shape_dtw_alignment_path(
            x=_x, y=_y, window=window, descriptor=descriptor, reach=reach
        )
    if x.ndim == 2 and y.ndim == 2:
        x_pad = np.pad(x, [[0, 0], [reach, reach]], mode="edge")
        y_pad = np.pad(y, [[0, 0], [reach, reach]], mode="edge")

        return _shape_dtw_alignment_path(
            x=x_pad, y=y_pad, window=window, descriptor=descriptor, reach=reach
        )

    raise ValueError("x and y must be 1D or 2D")


def _shape_dtw_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    descriptor: str = "identity",
    reach: int = 30,
) -> List[Tuple[int, int]]:
    """Compute the ShapeDTW alignment path between two series x and y.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
        is used.
    descriptor : str, default=None (if None then identity is used).
        This defines which transformation is applied on the sub-sequences.
    reach : int, default=30.
        This is the length of the sub-sequences.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
    """
    shapedtw_cost_mat = _shape_dtw_cost_matrix(
        x=x, y=y, window=window, descriptor=descriptor, reach=reach
    )

    return compute_min_return_path(shapedtw_cost_mat)


def shape_dtw_pairwise_distance(
    X: np.ndarray,
    y: np.ndarray = None,
    window: float = None,
    descriptor: str = "identity",
    reach: int = 30,
) -> np.ndarray:
    """Compute the ShapeDTW pairwise distance among a set of series.

    Parameters
    ----------
    X : np.ndarray
        A set of time series, either univariate, shape ``(n_instances, n_timepoints,)``,
        or multivariate, shape ``(n_instances, n_channels, n_timepoints)``.
    y : np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_instances, m_timepoints)`` or ``(m_instances, m_channels, m_timepoints)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
        is used.
    descriptor : str, default=None (if None then identity is used).
        This defines which transformation is applied on the sub-sequences.
    reach : int, default=30.
        This is the length of the sub-sequences.

    Returns
    -------
    np.ndarray
        ShapeDTW pairwise matrix between the instances of X of shape
        ``(n_instances, n_instances)`` or between X and y of shape ``(n_instances,
        n_instances)``.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
    """
    if y is None:
        if X.ndim == 3:
            X = np.pad(X, [[0, 0], [0, 0], [reach, reach]])
            return _shape_dtw_pairwise_distance(
                X,
                window,
            )
        if X.ndim == 2:
            X = np.pad(X, [[0, 0], [reach, reach]])
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _shape_dtw_pairwise_distance(
                _X,
                window,
            )
        raise ValueError("X must be 2D or 3D arrays")
    else:
        if X.ndim == 3 and y.ndim == 2:
            X = np.pad(X, [[0, 0], [0, 0], [reach, reach]])
            y = np.pad(y, [[0, 0], [reach, reach]])
            y = np.reshape(y, (y.shape[0], 1, y.shape[1]))
            return _shape_dtw_pairwise_distance(
                X=X, y=y, window=window, descriptor=descriptor, reach=reach
            )
        elif X.ndim == 3 and y.ndim == 3:
            X = np.pad(X, [[0, 0], [0, 0], [reach, reach]])
            y = np.pad(y, [[0, 0], [0, 0], [reach, reach]])
            return _shape_dtw_pairwise_distance(
                X=X, y=y, window=window, descriptor=descriptor, reach=reach
            )
        elif X.ndim == 2 and y.ndim == 2:
            X = np.pad(X, [[0, 0], [reach, reach]])
            X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
            y = np.pad(y, [[0, 0], [reach, reach]])
            y = np.reshape(y, (y.shape[0], 1, y.shape[1]))
            return _shape_dtw_pairwise_distance(
                X=X, y=y, window=window, descriptor=descriptor, reach=reach
            )
        elif X.ndim == 2 and y.ndim == 3:
            X = np.pad(X, [[0, 0], [reach, reach]])
            X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
            y = np.pad(y, [[0, 0], [0, 0], [reach, reach]])
            return _shape_dtw_pairwise_distance(
                X=X, y=y, window=window, descriptor=descriptor, reach=reach
            )


@njit(fastmath=True)
def _shape_dtw_pairwise_distance(
    X: np.ndarray,
    y: np.ndarray = None,
    window: float = None,
    descriptor: str = "identity",
    reach: int = 30,
) -> np.ndarray:
    """Compute the ShapeDTW pairwise distance among a set of series.

    Parameters
    ----------
    X : np.ndarray
        A set of time series, either univariate, shape ``(n_instances, n_timepoints,)``,
        or multivariate, shape ``(n_instances, n_channels, n_timepoints)``.
    y : np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_instances, m_timepoints)`` or ``(m_instances, m_channels, m_timepoints)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
        is used.
    descriptor : str, default=None (if None then identity is used).
        This defines which transformation is applied on the sub-sequences.
    reach : int, default=30.
        This is the length of the sub-sequences.

    Returns
    -------
    np.ndarray
        ShapeDTW pairwise matrix between the instances of X of shape
        ``(n_instances, n_instances)`` or between X and y of shape ``(n_instances,
        n_instances)``.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
    """
    if y is None:
        y = np.copy(X)

    distances = np.zeros(shape=(len(X), len(y)))

    for i in range(len(X)):
        for j in range(len(y)):
            distances[i, j] = shape_dtw_distance(
                x=X[i], y=y[i], descriptor=descriptor, window=window, reach=reach
            )

    return distances
