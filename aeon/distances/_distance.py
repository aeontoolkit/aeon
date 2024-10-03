__maintainer__ = []

from typing import Any, Callable, Optional, TypedDict, Union

import numpy as np
from typing_extensions import Unpack

from aeon.distances._adtw import (
    adtw_alignment_path,
    adtw_cost_matrix,
    adtw_distance,
    adtw_pairwise_distance,
)
from aeon.distances._ddtw import (
    ddtw_alignment_path,
    ddtw_cost_matrix,
    ddtw_distance,
    ddtw_pairwise_distance,
)
from aeon.distances._dtw import (
    dtw_alignment_path,
    dtw_cost_matrix,
    dtw_distance,
    dtw_pairwise_distance,
)
from aeon.distances._edr import (
    edr_alignment_path,
    edr_cost_matrix,
    edr_distance,
    edr_pairwise_distance,
)
from aeon.distances._erp import (
    erp_alignment_path,
    erp_cost_matrix,
    erp_distance,
    erp_pairwise_distance,
)
from aeon.distances._euclidean import euclidean_distance, euclidean_pairwise_distance
from aeon.distances._lcss import (
    lcss_alignment_path,
    lcss_cost_matrix,
    lcss_distance,
    lcss_pairwise_distance,
)
from aeon.distances._mahalanobis import (
    mahalanobis_distance,
    mahalanobis_pairwise_distance,
)
from aeon.distances._manhattan import manhattan_distance, manhattan_pairwise_distance
from aeon.distances._minkowski import minkowski_distance, minkowski_pairwise_distance
from aeon.distances._mpdist import mpdist
from aeon.distances._msm import (
    msm_alignment_path,
    msm_cost_matrix,
    msm_distance,
    msm_pairwise_distance,
)
from aeon.distances._sbd import sbd_distance, sbd_pairwise_distance
from aeon.distances._shape_dtw import (
    shape_dtw_alignment_path,
    shape_dtw_cost_matrix,
    shape_dtw_distance,
    shape_dtw_pairwise_distance,
)
from aeon.distances._squared import squared_distance, squared_pairwise_distance
from aeon.distances._twe import (
    twe_alignment_path,
    twe_cost_matrix,
    twe_distance,
    twe_pairwise_distance,
)
from aeon.distances._utils import _convert_to_list, _is_multivariate
from aeon.distances._wddtw import (
    wddtw_alignment_path,
    wddtw_cost_matrix,
    wddtw_distance,
    wddtw_pairwise_distance,
)
from aeon.distances._wdtw import (
    wdtw_alignment_path,
    wdtw_cost_matrix,
    wdtw_distance,
    wdtw_pairwise_distance,
)


class DistanceKwargs(TypedDict, total=False):
    window: Optional[float]
    itakura_max_slope: Optional[float]
    p: float
    w: np.ndarray
    g: float
    descriptor: str
    reach: int
    epsilon: float
    g_arr: np.ndarray
    nu: float
    lmbda: float
    independent: bool
    c: float
    warp_penalty: float
    standardize: bool
    m: int


DistanceFunction = Callable[[np.ndarray, np.ndarray, Any], float]
AlignmentPathFunction = Callable[
    [np.ndarray, np.ndarray, Any], tuple[list[tuple[int, int]], float]
]
CostMatrixFunction = Callable[[np.ndarray, np.ndarray, Any], np.ndarray]
PairwiseFunction = Callable[[np.ndarray, np.ndarray, Any], np.ndarray]


def distance(
    x: np.ndarray,
    y: np.ndarray,
    metric: Union[str, DistanceFunction],
    **kwargs: Unpack[DistanceKwargs],
) -> float:
    """Compute the distance between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    metric : str or Callable
        The distance metric to use.
        A list of valid distance metrics can be found in the documentation for
        :func:`aeon.distances.get_distance_function` or by calling  the function
        :func:`aeon.distances.get_distance_function_names`.
    kwargs : Any
        Arguments for metric. Refer to each metrics documentation for a list of
        possible arguments.

    Returns
    -------
    float
        Distance between the x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, or 2D arrays.
        If metric is not a valid string or callable.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> distance(x, y, metric="dtw")
    768.0
    """
    if metric == "squared":
        return squared_distance(x, y)
    elif metric == "euclidean":
        return euclidean_distance(x, y)
    elif metric == "manhattan":
        return manhattan_distance(x, y)
    elif metric == "minkowski":
        return minkowski_distance(x, y, kwargs.get("p", 2.0), kwargs.get("w", None))
    elif metric == "dtw":
        return dtw_distance(x, y, kwargs.get("window"), kwargs.get("itakura_max_slope"))
    elif metric == "ddtw":
        return ddtw_distance(
            x, y, kwargs.get("window"), kwargs.get("itakura_max_slope")
        )
    elif metric == "wdtw":
        return wdtw_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("g", 0.05),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "shape_dtw":
        return shape_dtw_distance(
            x,
            y,
            window=kwargs.get("window"),
            itakura_max_slope=kwargs.get("itakura_max_slope"),
            descriptor=kwargs.get("descriptor", "identity"),
            reach=kwargs.get("reach", 30),
            transformation_precomputed=kwargs.get("transformation_precomputed", False),
            transformed_x=kwargs.get("transformed_x", None),
            transformed_y=kwargs.get("transformed_y", None),
        )
    elif metric == "wddtw":
        return wddtw_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("g", 0.05),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "lcss":
        return lcss_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("epsilon", 1.0),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "erp":
        return erp_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("g", 0.0),
            kwargs.get("g_arr", None),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "edr":
        return edr_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("epsilon"),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "twe":
        return twe_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("nu", 0.001),
            kwargs.get("lmbda", 1.0),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "msm":
        return msm_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("independent", True),
            kwargs.get("c", 1.0),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "mpdist":
        return mpdist(x, y, kwargs.get("m", 0))
    elif metric == "adtw":
        return adtw_distance(
            x,
            y,
            itakura_max_slope=kwargs.get("itakura_max_slope"),
            window=kwargs.get("window"),
            warp_penalty=kwargs.get("warp_penalty", 1.0),
        )
    elif metric == "sbd":
        return sbd_distance(x, y, kwargs.get("standardize", True))
    elif metric == "mahalanobis":
        return mahalanobis_distance(x, y, kwargs.get("m"))
    else:
        if isinstance(metric, Callable):
            return metric(x, y, **kwargs)
        raise ValueError("Metric must be one of the supported strings or a callable")


def pairwise_distance(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    metric: Union[str, DistanceFunction, None] = None,
    **kwargs: Unpack[DistanceKwargs],
) -> np.ndarray:
    """Compute the pairwise distance matrix between two time series.

    Parameters
    ----------
    X : np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
         or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or None, default=None
       A single series or a collection of time series of shape ``(m_timepoints,)`` or
       ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``
    metric : str or Callable
        The distance metric to use.
        A list of valid distance metrics can be found in the documentation for
        :func:`aeon.distances.get_distance_function` or by calling  the function
        :func:`aeon.distances.get_distance_function_names`.
    kwargs : Any
        Extra arguments for metric. Refer to each metric documentation for a list of
        possible arguments.

    Returns
    -------
    np.ndarray (n_cases, n_cases)
        pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.
        If metric is not a valid string or callable.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> pairwise_distance(X, metric='dtw')
    array([[  0.,  26., 108.],
           [ 26.,   0.,  26.],
           [108.,  26.,   0.]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> pairwise_distance(X, y, metric='dtw')
    array([[300., 507., 768.],
           [147., 300., 507.],
           [ 48., 147., 300.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([11, 12, 13])
    >>> pairwise_distance(X, y_univariate, metric='dtw')
    array([[300.],
           [147.],
           [ 48.]])
    """
    if metric == "squared":
        return squared_pairwise_distance(x, y)
    elif metric == "euclidean":
        return euclidean_pairwise_distance(x, y)
    elif metric == "manhattan":
        return manhattan_pairwise_distance(x, y)
    elif metric == "minkowski":
        return minkowski_pairwise_distance(
            x, y, kwargs.get("p", 2.0), kwargs.get("w", None)
        )
    elif metric == "dtw":
        return dtw_pairwise_distance(
            x, y, kwargs.get("window"), kwargs.get("itakura_max_slope")
        )
    elif metric == "shape_dtw":
        return shape_dtw_pairwise_distance(
            x,
            y,
            window=kwargs.get("window"),
            itakura_max_slope=kwargs.get("itakura_max_slope"),
            descriptor=kwargs.get("descriptor", "identity"),
            reach=kwargs.get("reach", 30),
            transformation_precomputed=kwargs.get("transformation_precomputed", False),
            transformed_x=kwargs.get("transformed_x", None),
            transformed_y=kwargs.get("transformed_y", None),
        )
    elif metric == "ddtw":
        return ddtw_pairwise_distance(
            x, y, kwargs.get("window"), kwargs.get("itakura_max_slope")
        )
    elif metric == "wdtw":
        return wdtw_pairwise_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("g", 0.05),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "wddtw":
        return wddtw_pairwise_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("g", 0.05),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "lcss":
        return lcss_pairwise_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("epsilon", 1.0),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "erp":
        return erp_pairwise_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("g", 0.0),
            kwargs.get("g_arr", None),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "edr":
        return edr_pairwise_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("epsilon"),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "twe":
        return twe_pairwise_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("nu", 0.001),
            kwargs.get("lmbda", 1.0),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "msm":
        return msm_pairwise_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("independent", True),
            kwargs.get("c", 1.0),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "mpdist":
        return _custom_func_pairwise(x, y, mpdist, **kwargs)
    elif metric == "adtw":
        return adtw_pairwise_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("itakura_max_slope"),
            kwargs.get("warp_penalty", 1.0),
        )
    elif metric == "sbd":
        return sbd_pairwise_distance(x, y, kwargs.get("standardize", True))
    elif metric == "mahalanobis":
        return mahalanobis_pairwise_distance(x, y, kwargs.get("m"))
    else:
        if isinstance(metric, Callable):
            return _custom_func_pairwise(x, y, metric, **kwargs)
        raise ValueError("Metric must be one of the supported strings or a callable")


def _custom_func_pairwise(
    X: Optional[Union[np.ndarray, list[np.ndarray]]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    dist_func: Union[DistanceFunction, None] = None,
    **kwargs: Unpack[DistanceKwargs],
) -> np.ndarray:
    if dist_func is None:
        raise ValueError("dist_func must be a callable")

    multivariate_conversion = _is_multivariate(X, y)
    X, _ = _convert_to_list(X, "X", multivariate_conversion)
    if y is None:
        # To self
        return _custom_pairwise_distance(X, dist_func, **kwargs)
    y, _ = _convert_to_list(y, "y", multivariate_conversion)
    return _custom_from_multiple_to_multiple_distance(X, y, dist_func, **kwargs)


def _custom_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    dist_func: DistanceFunction,
    **kwargs: Unpack[DistanceKwargs],
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            distances[i, j] = dist_func(X[i], X[j], **kwargs)
            distances[j, i] = distances[i, j]

    return distances


def _custom_from_multiple_to_multiple_distance(
    x: Union[np.ndarray, list[np.ndarray]],
    y: Union[np.ndarray, list[np.ndarray]],
    dist_func: DistanceFunction,
    **kwargs: Unpack[DistanceKwargs],
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    for i in range(n_cases):
        for j in range(m_cases):
            distances[i, j] = dist_func(x[i], y[j], **kwargs)
    return distances


def alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    metric: str,
    **kwargs: Unpack[DistanceKwargs],
) -> tuple[list[tuple[int, int]], float]:
    """Compute the alignment path and distance between two time series.

    Parameters
    ----------
    x : np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y : np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    metric : str
        The distance metric to use.
        A list of valid distance metrics can be found in the documentation for
        :func:`aeon.distances.get_distance_function` or by calling  the function
        :func:`aeon.distances.get_distance_function_names`.
    kwargs : any
        Arguments for metric. Refer to each metrics documentation for a list of
        possible arguments.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The dtw distance betweeen the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D, or 2D arrays.
        If metric is not one of the supported strings or a callable.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> alignment_path(x, y, metric='dtw')
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 4.0)
    """
    if metric == "dtw":
        return dtw_alignment_path(
            x, y, kwargs.get("window"), kwargs.get("itakura_max_slope")
        )
    elif metric == "shape_dtw":
        return shape_dtw_alignment_path(
            x,
            y,
            window=kwargs.get("window"),
            itakura_max_slope=kwargs.get("itakura_max_slope"),
            descriptor=kwargs.get("descriptor", "identity"),
            reach=kwargs.get("reach", 30),
            transformation_precomputed=kwargs.get("transformation_precomputed", False),
            transformed_x=kwargs.get("transformed_x", None),
            transformed_y=kwargs.get("transformed_y", None),
        )
    elif metric == "ddtw":
        return ddtw_alignment_path(
            x, y, kwargs.get("window"), kwargs.get("itakura_max_slope")
        )
    elif metric == "wdtw":
        return wdtw_alignment_path(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("g", 0.05),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "wddtw":
        return wddtw_alignment_path(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("g", 0.05),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "lcss":
        return lcss_alignment_path(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("epsilon", 1.0),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "erp":
        return erp_alignment_path(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("g", 0.0),
            kwargs.get("g_arr", None),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "edr":
        return edr_alignment_path(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("epsilon"),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "twe":
        return twe_alignment_path(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("nu", 0.001),
            kwargs.get("lmbda", 1.0),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "msm":
        return msm_alignment_path(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("independent", True),
            kwargs.get("c", 1.0),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "adtw":
        return adtw_alignment_path(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("itakura_max_slope"),
            kwargs.get("warp_penalty", 1.0),
        )
    else:
        raise ValueError("Metric must be one of the supported strings")


def cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    metric: str,
    **kwargs: Unpack[DistanceKwargs],
) -> np.ndarray:
    """Compute the alignment path and distance between two time series.

    Parameters
    ----------
    x : np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y : np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    metric : str or Callable
        The distance metric to use.
        A list of valid distance metrics can be found in the documentation for
        :func:`aeon.distances.get_distance_function` or by calling  the function
        :func:`aeon.distances.get_distance_function_names`.
    kwargs : Any
        Arguments for metric. Refer to each metrics documentation for a list of
        possible arguments.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, or 2D arrays.
        If metric is not one of the supported strings or a callable.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> cost_matrix(x, y, metric="dtw")
    array([[  0.,   1.,   5.,  14.,  30.,  55.,  91., 140., 204., 285.],
           [  1.,   0.,   1.,   5.,  14.,  30.,  55.,  91., 140., 204.],
           [  5.,   1.,   0.,   1.,   5.,  14.,  30.,  55.,  91., 140.],
           [ 14.,   5.,   1.,   0.,   1.,   5.,  14.,  30.,  55.,  91.],
           [ 30.,  14.,   5.,   1.,   0.,   1.,   5.,  14.,  30.,  55.],
           [ 55.,  30.,  14.,   5.,   1.,   0.,   1.,   5.,  14.,  30.],
           [ 91.,  55.,  30.,  14.,   5.,   1.,   0.,   1.,   5.,  14.],
           [140.,  91.,  55.,  30.,  14.,   5.,   1.,   0.,   1.,   5.],
           [204., 140.,  91.,  55.,  30.,  14.,   5.,   1.,   0.,   1.],
           [285., 204., 140.,  91.,  55.,  30.,  14.,   5.,   1.,   0.]])
    """
    if metric == "dtw":
        return dtw_cost_matrix(
            x, y, kwargs.get("window"), kwargs.get("itakura_max_slope")
        )
    elif metric == "shape_dtw":
        return shape_dtw_cost_matrix(
            x,
            y,
            window=kwargs.get("window"),
            itakura_max_slope=kwargs.get("itakura_max_slope"),
            descriptor=kwargs.get("descriptor", "identity"),
            reach=kwargs.get("reach", 30),
            transformation_precomputed=kwargs.get("transformation_precomputed", False),
            transformed_x=kwargs.get("transformed_x", None),
            transformed_y=kwargs.get("transformed_y", None),
        )
    elif metric == "ddtw":
        return ddtw_cost_matrix(
            x, y, kwargs.get("window"), kwargs.get("itakura_max_slope")
        )
    elif metric == "wdtw":
        return wdtw_cost_matrix(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("g", 0.05),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "wddtw":
        return wddtw_cost_matrix(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("g", 0.05),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "lcss":
        return lcss_cost_matrix(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("epsilon", 1.0),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "erp":
        return erp_cost_matrix(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("g", 0.0),
            kwargs.get("g_arr", None),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "edr":
        return edr_cost_matrix(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("epsilon"),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "twe":
        return twe_cost_matrix(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("nu", 0.001),
            kwargs.get("lmbda", 1.0),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "msm":
        return msm_cost_matrix(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("independent", True),
            kwargs.get("c", 1.0),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "adtw":
        return adtw_cost_matrix(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("itakura_max_slope"),
            kwargs.get("warp_penalty", 1.0),
        )
    else:
        raise ValueError("Metric must be one of the supported strings")


def get_distance_function_names() -> list[str]:
    """Get a list of distance function names in aeon.

    All distance function names have two associated functions:
        name_distance
        name_pairwise_distance
    Elastic distances have two additional functions associated with them:
        name_alignment_path
        name_cost_matrix

    Returns
    -------
    List[str]
        List of distance function names in aeon.

    Examples
    --------
    >>> from aeon.distances import get_distance_function_names
    >>> names = get_distance_function_names()
    >>> names[0]
    'adtw'

    """
    return sorted(DISTANCES_DICT.keys())


def get_distance_function(metric: Union[str, DistanceFunction]) -> DistanceFunction:
    """Get the distance function for a given metric string or callable.

    =============== ========================================
    metric          Distance Function
    =============== ========================================
    'dtw'           distances.dtw_distance
    'shape_dtw'     distances.shape_dtw_distance
    'ddtw'          distances.ddtw_distance
    'wdtw'          distances.wdtw_distance
    'wddtw'         distances.wddtw_distance
    'adtw'          distances.adtw_distance
    'erp'           distances.erp_distance
    'edr'           distances.edr_distance
    'msm'           distances.msm_distance
    'twe'           distances.twe_distance
    'lcss'          distances.lcss_distance
    'euclidean'     distances.euclidean_distance
    'squared'       distances.squared_distance
    'manhattan'     distances.manhattan_distance
    'minkowski'     distances.minkowski_distance
    'sbd'           distances.sbd_distance
    'mahalanobis'   distances.mahalanobis_distance
    =============== ========================================

    Parameters
    ----------
    metric : str or Callable
        The distance metric to use.
        If string given then it will be resolved to a alignment path function.
        If a callable is given, the value must be a function that accepts two
        numpy arrays and **kwargs returns a float.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, Any], float]
        The distance function for the given metric.

    Raises
    ------
    ValueError
        If metric is not one of the supported strings or a callable.

    Examples
    --------
    >>> from aeon.distances import get_distance_function
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> dtw_dist_func = get_distance_function("dtw")
    >>> dtw_dist_func(x, y, window=0.2)
    874.0
    """
    return _resolve_key_from_distance(metric, "distance")


def get_pairwise_distance_function(
    metric: Union[str, PairwiseFunction]
) -> PairwiseFunction:
    """Get the pairwise distance function for a given metric string or callable.

    =============== ========================================
    metric          Distance Function
    =============== ========================================
    'dtw'           distances.dtw_pairwise_distance
    'shape_dtw'     distances.shape_dtw_pairwise_distance
    'ddtw'          distances.ddtw_pairwise_distance
    'wdtw'          distances.wdtw_pairwise_distance
    'wddtw'         distances.wddtw_pairwise_distance
    'adtw'          distances.adtw_pairwise_distance
    'erp'           distances.erp_pairwise_distance
    'edr'           distances.edr_pairwise_distance
    'msm'           distances.msm_pairiwse_distance
    'twe'           distances.twe_pairwise_distance
    'lcss'          distances.lcss_pairwise_distance
    'euclidean'     distances.euclidean_pairwise_distance
    'squared'       distances.squared_pairwise_distance
    'manhattan'     distances.manhattan_pairwise_distance
    'minkowski'     distances.minkowski_pairwise_distance
    'sbd'           distances.sbd_pairwise_distance
    'mahalanobis'   distances.mahalanobis_pairwise_distance
    =============== ========================================

    Parameters
    ----------
    metric : str or Callable
        The metric string to resolve to a alignment path function.
        If string given then it will be resolved to a alignment path function.
        If a callable is given, the value must be a function that accepts two
        numpy arrays and **kwargs returns a np.ndarray that is the pairwise distance
        between each time series.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, Any], np.ndarray]
        The pairwise distance function for the given metric.

    Raises
    ------
    ValueError
        If metric is not one of the supported strings or a callable.

    Examples
    --------
    >>> from aeon.distances import get_pairwise_distance_function
    >>> import numpy as np
    >>> x = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> dtw_pairwise_dist_func = get_pairwise_distance_function("dtw")
    >>> dtw_pairwise_dist_func(x, y, window=0.2)
    array([[300., 507., 768.],
           [147., 300., 507.],
           [ 48., 147., 300.]])
    """
    return _resolve_key_from_distance(metric, "pairwise_distance")


def get_alignment_path_function(metric: str) -> AlignmentPathFunction:
    """Get the alignment path function for a given metric string or callable.

    =============== ========================================
    metric          Distance Function
    =============== ========================================
    'dtw'           distances.dtw_alignment_path
    'shape_dtw'     distances.shape_dtw_alignment_path
    'ddtw'          distances.ddtw_alignment_path
    'wdtw'          distances.wdtw_alignment_path
    'wddtw'         distances.wddtw_alignment_path
    'adtw'          distances.adtw_alignment_path
    'erp'           distances.erp_alignment_path
    'edr'           distances.edr_alignment_path
    'msm'           distances.msm_alignment_path
    'twe'           distances.twe_alignment_path
    'lcss'          distances.lcss_alignment_path
    =============== ========================================

    Parameters
    ----------
    metric : str or Callable
        The metric string to resolve to a alignment path function.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, Any], Tuple[List[Tuple[int, int]], float]]
        The alignment path function for the given metric.

    Raises
    ------
    ValueError
        If metric is not one of the supported strings or a callable.
        If the metric doesn't have an alignment path function.

    Examples
    --------
    >>> from aeon.distances import get_alignment_path_function
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 4, 5]])
    >>> y = np.array([[11, 12, 13, 14, 15]])
    >>> dtw_alignment_path_func = get_alignment_path_function("dtw")
    >>> dtw_alignment_path_func(x, y, window=0.2)
    ([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)], 500.0)
    """
    return _resolve_key_from_distance(metric, "alignment_path")


def get_cost_matrix_function(metric: str) -> CostMatrixFunction:
    """Get the cost matrix function for a given metric string or callable.

    =============== ========================================
    metric          Distance Function
    =============== ========================================
    'dtw'           distances.dtw_cost_matrix
    'shape_dtw'     distances.shape_dtw_cost_matrix
    'ddtw'          distances.ddtw_cost_matrix
    'wdtw'          distances.wdtw_cost_matrix
    'wddtw'         distances.wddtw_cost_matrix
    'adtw'          distances.adtw_cost_matrix
    'erp'           distances.erp_cost_matrix
    'edr'           distances.edr_cost_matrix
    'msm'           distances.msm_cost_matrix
    'twe'           distances.twe_cost_matrix
    'lcss'          distances.lcss_cost_matrix
    =============== ========================================

    Parameters
    ----------
    metric : str or Callable
        The metric string to resolve to a cost matrix function.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, Any], np.ndarray]
        The cost matrix function for the given metric.

    Raises
    ------
    ValueError
        If metric is not one of the supported strings or a callable.
        If the metric doesn't have a cost matrix function.

    Examples
    --------
    >>> from aeon.distances import get_cost_matrix_function
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 4, 5]])
    >>> y = np.array([[11, 12, 13, 14, 15]])
    >>> dtw_cost_matrix_func = get_cost_matrix_function("dtw")
    >>> dtw_cost_matrix_func(x, y, window=0.2)
    array([[100., 221.,  inf,  inf,  inf],
           [181., 200., 321.,  inf,  inf],
           [ inf, 262., 300., 421.,  inf],
           [ inf,  inf, 343., 400., 521.],
           [ inf,  inf,  inf, 424., 500.]])
    """
    return _resolve_key_from_distance(metric, "cost_matrix")


def _resolve_key_from_distance(metric: Union[str, Callable], key: str) -> Any:
    if isinstance(metric, Callable):
        return metric
    if metric == "mpdist":
        return mpdist
    dist = DISTANCES_DICT.get(metric)
    if dist is None:
        raise ValueError(f"Unknown metric {metric}")
    dist_callable = dist.get(key)
    if dist_callable is None:
        raise ValueError(f"Metric {metric} does not have a {key} function")
    return dist_callable


DISTANCES = [
    {
        "name": "euclidean",
        "distance": euclidean_distance,
        "pairwise_distance": euclidean_pairwise_distance,
    },
    {
        "name": "squared",
        "distance": squared_distance,
        "pairwise_distance": squared_pairwise_distance,
    },
    {
        "name": "manhattan",
        "distance": manhattan_distance,
        "pairwise_distance": manhattan_pairwise_distance,
    },
    {
        "name": "minkowski",
        "distance": minkowski_distance,
        "pairwise_distance": minkowski_pairwise_distance,
    },
    {
        "name": "dtw",
        "distance": dtw_distance,
        "pairwise_distance": dtw_pairwise_distance,
        "cost_matrix": dtw_cost_matrix,
        "alignment_path": dtw_alignment_path,
    },
    {
        "name": "ddtw",
        "distance": ddtw_distance,
        "pairwise_distance": ddtw_pairwise_distance,
        "cost_matrix": ddtw_cost_matrix,
        "alignment_path": ddtw_alignment_path,
    },
    {
        "name": "wdtw",
        "distance": wdtw_distance,
        "pairwise_distance": wdtw_pairwise_distance,
        "cost_matrix": wdtw_cost_matrix,
        "alignment_path": wdtw_alignment_path,
    },
    {
        "name": "wddtw",
        "distance": wddtw_distance,
        "pairwise_distance": wddtw_pairwise_distance,
        "cost_matrix": wddtw_cost_matrix,
        "alignment_path": wddtw_alignment_path,
    },
    {
        "name": "lcss",
        "distance": lcss_distance,
        "pairwise_distance": lcss_pairwise_distance,
        "cost_matrix": lcss_cost_matrix,
        "alignment_path": lcss_alignment_path,
    },
    {
        "name": "erp",
        "distance": erp_distance,
        "pairwise_distance": erp_pairwise_distance,
        "cost_matrix": erp_cost_matrix,
        "alignment_path": erp_alignment_path,
    },
    {
        "name": "edr",
        "distance": edr_distance,
        "pairwise_distance": edr_pairwise_distance,
        "cost_matrix": edr_cost_matrix,
        "alignment_path": edr_alignment_path,
    },
    {
        "name": "twe",
        "distance": twe_distance,
        "pairwise_distance": twe_pairwise_distance,
        "cost_matrix": twe_cost_matrix,
        "alignment_path": twe_alignment_path,
    },
    {
        "name": "msm",
        "distance": msm_distance,
        "pairwise_distance": msm_pairwise_distance,
        "cost_matrix": msm_cost_matrix,
        "alignment_path": msm_alignment_path,
    },
    {
        "name": "adtw",
        "distance": adtw_distance,
        "pairwise_distance": adtw_pairwise_distance,
        "cost_matrix": adtw_cost_matrix,
        "alignment_path": adtw_alignment_path,
    },
    {
        "name": "shape_dtw",
        "distance": shape_dtw_distance,
        "pairwise_distance": shape_dtw_pairwise_distance,
        "cost_matrix": shape_dtw_cost_matrix,
        "alignment_path": shape_dtw_alignment_path,
    },
    {
        "name": "sbd",
        "distance": sbd_distance,
        "pairwise_distance": sbd_pairwise_distance,
    },
    {
        "name": "mahalanobis",
        "distance": mahalanobis_distance,
        "pairwise_distance": mahalanobis_pairwise_distance,
    },
]

DISTANCES_DICT = {d["name"]: d for d in DISTANCES}
