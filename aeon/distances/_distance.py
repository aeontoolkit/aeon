__maintainer__ = []

from enum import Enum
from typing import Any, Callable, Optional, TypedDict, Union

import numpy as np
from typing_extensions import Unpack

from aeon.distances._mpdist import mp_distance, mp_pairwise_distance
from aeon.distances._sbd import sbd_distance, sbd_pairwise_distance
from aeon.distances._shift_scale_invariant import (
    shift_scale_invariant_distance,
    shift_scale_invariant_pairwise_distance,
)
from aeon.distances.elastic import (
    adtw_alignment_path,
    adtw_cost_matrix,
    adtw_distance,
    adtw_pairwise_distance,
    ddtw_alignment_path,
    ddtw_cost_matrix,
    ddtw_distance,
    ddtw_pairwise_distance,
    dtw_alignment_path,
    dtw_cost_matrix,
    dtw_distance,
    dtw_pairwise_distance,
    edr_alignment_path,
    edr_cost_matrix,
    edr_distance,
    edr_pairwise_distance,
    erp_alignment_path,
    erp_cost_matrix,
    erp_distance,
    erp_pairwise_distance,
    lcss_alignment_path,
    lcss_cost_matrix,
    lcss_distance,
    lcss_pairwise_distance,
    msm_alignment_path,
    msm_cost_matrix,
    msm_distance,
    msm_pairwise_distance,
    shape_dtw_alignment_path,
    shape_dtw_cost_matrix,
    shape_dtw_distance,
    shape_dtw_pairwise_distance,
    soft_dtw_alignment_path,
    soft_dtw_cost_matrix,
    soft_dtw_distance,
    soft_dtw_pairwise_distance,
    swale_distance,
    swale_pairwise_distance,
    twe_alignment_path,
    twe_cost_matrix,
    twe_distance,
    twe_pairwise_distance,
    wddtw_alignment_path,
    wddtw_cost_matrix,
    wddtw_distance,
    wddtw_pairwise_distance,
    wdtw_alignment_path,
    wdtw_cost_matrix,
    wdtw_distance,
    wdtw_pairwise_distance,
)
from aeon.distances.mindist import (
    mindist_dft_sfa_distance,
    mindist_dft_sfa_pairwise_distance,
    mindist_paa_sax_distance,
    mindist_paa_sax_pairwise_distance,
    mindist_sax_distance,
    mindist_sax_pairwise_distance,
    mindist_sfa_distance,
    mindist_sfa_pairwise_distance,
)
from aeon.distances.pointwise import (
    euclidean_distance,
    euclidean_pairwise_distance,
    manhattan_distance,
    manhattan_pairwise_distance,
    minkowski_distance,
    minkowski_pairwise_distance,
    squared_distance,
    squared_pairwise_distance,
)
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


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
    max_shift: Optional[int]
    gamma: float


DistanceFunction = Callable[[np.ndarray, np.ndarray, Any], float]
AlignmentPathFunction = Callable[
    [np.ndarray, np.ndarray, Any], tuple[list[tuple[int, int]], float]
]
CostMatrixFunction = Callable[[np.ndarray, np.ndarray, Any], np.ndarray]
PairwiseFunction = Callable[[np.ndarray, np.ndarray, Any], np.ndarray]


def distance(
    x: np.ndarray,
    y: np.ndarray,
    method: Union[str, DistanceFunction],
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
    method : str or Callable
        The distance to use.
        A list of valid distance can be found in the documentation for
        :func:`aeon.distances.get_distance_function` or by calling  the function
        :func:`aeon.distances.get_distance_function_names`.
    kwargs : Any
        Arguments for distance. Refer to each distance documentation for a list of
        possible arguments.

    Returns
    -------
    float
        Distance between the x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, or 2D arrays.
        If distance is not a valid string or callable.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> distance(x, y, method="dtw")
    768.0
    """
    if method in DISTANCES_DICT:
        return DISTANCES_DICT[method]["distance"](x, y, **kwargs)
    elif isinstance(method, Callable):
        return method(x, y, **kwargs)
    else:
        raise ValueError("Method must be one of the supported strings or a callable")


def pairwise_distance(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    method: Union[str, DistanceFunction, None] = None,
    symmetric: bool = True,
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
    method : str or Callable
        The distance to use.
        A list of valid distance can be found in the documentation for
        :func:`aeon.distances.get_distance_function` or by calling  the function
        :func:`aeon.distances.get_distance_function_names`.
    symmetric : bool, default=True
        If True and a function is provided as the "method" paramter, then it will
        compute a symmetric distance matrix where d(x, y) = d(y, x). Only the lower
        triangle is calculated, and the upper triangle is ignored. If False and a
        function is provided as the "method" parameter, then it will compute an
        asymmetric distance matrix, and the entire matrix (including both upper and
        lower triangles) is returned.
    kwargs : Any
        Extra arguments for distance. Refer to each distance documentation for a list of
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
        If distance is not a valid string or callable.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> pairwise_distance(X, method='dtw')
    array([[  0.,  26., 108.],
           [ 26.,   0.,  26.],
           [108.,  26.,   0.]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> pairwise_distance(X, y, method='dtw')
    array([[300., 507., 768.],
           [147., 300., 507.],
           [ 48., 147., 300.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([11, 12, 13])
    >>> pairwise_distance(X, y_univariate, method='dtw')
    array([[300.],
           [147.],
           [ 48.]])
    """
    if method in PAIRWISE_DISTANCE:
        return DISTANCES_DICT[method]["pairwise_distance"](x, y, **kwargs)
    elif isinstance(method, Callable):
        if y is None and not symmetric:
            return _custom_func_pairwise(x, x, method, **kwargs)
        return _custom_func_pairwise(x, y, method, **kwargs)
    else:
        raise ValueError("Method must be one of the supported strings or a callable")


def _custom_func_pairwise(
    X: Optional[Union[np.ndarray, list[np.ndarray]]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    dist_func: Union[DistanceFunction, None] = None,
    **kwargs: Unpack[DistanceKwargs],
) -> np.ndarray:
    if dist_func is None:
        raise ValueError("dist_func must be a callable")

    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    X, _ = _convert_collection_to_numba_list(X, "X", multivariate_conversion)
    if y is None:
        # To self
        return _custom_pairwise_distance(X, dist_func, **kwargs)
    y, _ = _convert_collection_to_numba_list(y, "y", multivariate_conversion)
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
    method: Union[str, DistanceFunction, None] = None,
    **kwargs: Unpack[DistanceKwargs],
) -> tuple[list[tuple[int, int]], float]:
    """Compute the alignment path and distance between two time series.

    Parameters
    ----------
    x : np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y : np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    method : str or Callable
        The distance method to use.
        A list of valid distances can be found in the documentation for
        :func:`aeon.distances.get_distance_function` or by calling  the function
        :func:`aeon.distances.get_distance_function_names`.
    kwargs : any
        Arguments for distance. Refer to each distance documentation for a list of
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
        If distance is not one of the supported strings or a callable.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> alignment_path(x, y, method='dtw')
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 4.0)
    """
    if method in ALIGNMENT_PATH:
        return DISTANCES_DICT[method]["alignment_path"](x, y, **kwargs)
    elif isinstance(method, Callable):
        return method(x, y, **kwargs)
    else:
        raise ValueError("Method must be one of the supported strings")


def cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    method: Union[str, DistanceFunction, None] = None,
    **kwargs: Unpack[DistanceKwargs],
) -> np.ndarray:
    """Compute the alignment path and distance between two time series.

    Parameters
    ----------
    x : np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y : np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    method : str or Callable
        The distance to use.
        A list of valid distances can be found in the documentation for
        :func:`aeon.distances.get_distance_function` or by calling  the function
        :func:`aeon.distances.get_distance_function_names`.
    kwargs : Any
        Arguments for distance. Refer to each distance documentation for a list of
        possible arguments.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, or 2D arrays.
        If distance is not one of the supported strings or a callable.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> cost_matrix(x, y, method="dtw")
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
    if method in COST_MATRIX:
        return DISTANCES_DICT[method]["cost_matrix"](x, y, **kwargs)
    elif isinstance(method, Callable):
        return method(x, y, **kwargs)
    else:
        raise ValueError("Method must be one of the supported strings")


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


def get_distance_function(method: Union[str, DistanceFunction]) -> DistanceFunction:
    """Get the distance function for a given distance string or callable.

    =============== ========================================
    method          Distance Function
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
    'swale          distances.swale_distance
    'shift_scale'   distances.shift_scale_invariant_distance
    'soft_dtw'      distances.soft_dtw_distance
    =============== ========================================

    Parameters
    ----------
    method : str or Callable
        The distance to use.
        If string given then it will be resolved to a alignment path function.
        If a callable is given, the value must be a function that accepts two
        numpy arrays and **kwargs returns a float.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, Any], float]
        The distance function for the given method.

    Raises
    ------
    ValueError
        If distance is not one of the supported strings or a callable.

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
    return _resolve_key_from_distance(method, "distance")


def get_pairwise_distance_function(
    method: Union[str, PairwiseFunction]
) -> PairwiseFunction:
    """Get the pairwise distance function for a given method string or callable.

    =============== ========================================
    method          Distance Function
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
    'swale'         disrances.swale_pairwise_distance
    'shift_scale'   distances.shift_scale_invariant_pairwise_distance
    'soft_dtw'      distances.soft_dtw_pairwise_distance
    =============== ========================================

    Parameters
    ----------
    method : str or Callable
        The distance string to resolve to a alignment path function.
        If string given then it will be resolved to a alignment path function.
        If a callable is given, the value must be a function that accepts two
        numpy arrays and **kwargs returns a np.ndarray that is the pairwise distance
        between each time series.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, Any], np.ndarray]
        The pairwise distance function for the given method.

    Raises
    ------
    ValueError
        If mehtod is not one of the supported strings or a callable.

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
    return _resolve_key_from_distance(method, "pairwise_distance")


def get_alignment_path_function(method: str) -> AlignmentPathFunction:
    """Get the alignment path function for a given method string or callable.

    =============== ========================================
    method          Distance Function
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
    'soft_dtw'      distances.soft_dtw_alignment_path
    =============== ========================================

    Parameters
    ----------
    method : str or Callable
        The distance string to resolve to an alignment path function.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, Any], Tuple[List[Tuple[int, int]], float]]
        The alignment path function for the given distance.

    Raises
    ------
    ValueError
        If distance is not one of the supported strings or a callable.
        If the distance doesn't have an alignment path function.

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
    return _resolve_key_from_distance(method, "alignment_path")


def get_cost_matrix_function(method: str) -> CostMatrixFunction:
    """Get the cost matrix function for a given distance string or callable.

    =============== ========================================
    method          Distance Function
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
    'soft_dtw'      distances.soft_dtw_cost_matrix
    =============== ========================================

    Parameters
    ----------
    method : str or Callable
        The distance string to resolve to a cost matrix function.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, Any], np.ndarray]
        The cost matrix function for the given distance.

    Raises
    ------
    ValueError
        If distance is not one of the supported strings or a callable.
        If the distance doesn't have a cost matrix function.

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
    return _resolve_key_from_distance(method, "cost_matrix")


def _resolve_key_from_distance(method: Union[str, Callable], key: str) -> Any:
    if isinstance(method, Callable):
        return method
    if method == "mpdist":
        return mp_distance
    dist = DISTANCES_DICT.get(method)
    if dist is None:
        raise ValueError(f"Unknown method {method}")
    dist_callable = dist.get(key)
    if dist_callable is None:
        raise ValueError(f"Method {method} does not have a {key} function")
    return dist_callable


class DistanceType(Enum):
    """Enum for distance types."""

    POINTWISE = "pointwise"
    ELASTIC = "elastic"
    CROSS_CORRELATION = "cross-correlation"
    MIN_DISTANCE = "min-dist"
    MATRIX_PROFILE = "matrix-profile"


DISTANCES = [
    {
        "name": "euclidean",
        "distance": euclidean_distance,
        "pairwise_distance": euclidean_pairwise_distance,
        "type": DistanceType.POINTWISE,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "squared",
        "distance": squared_distance,
        "pairwise_distance": squared_pairwise_distance,
        "type": DistanceType.POINTWISE,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "manhattan",
        "distance": manhattan_distance,
        "pairwise_distance": manhattan_pairwise_distance,
        "type": DistanceType.POINTWISE,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "minkowski",
        "distance": minkowski_distance,
        "pairwise_distance": minkowski_pairwise_distance,
        "type": DistanceType.POINTWISE,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "dtw",
        "distance": dtw_distance,
        "pairwise_distance": dtw_pairwise_distance,
        "cost_matrix": dtw_cost_matrix,
        "alignment_path": dtw_alignment_path,
        "type": DistanceType.ELASTIC,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "ddtw",
        "distance": ddtw_distance,
        "pairwise_distance": ddtw_pairwise_distance,
        "cost_matrix": ddtw_cost_matrix,
        "alignment_path": ddtw_alignment_path,
        "type": DistanceType.ELASTIC,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "wdtw",
        "distance": wdtw_distance,
        "pairwise_distance": wdtw_pairwise_distance,
        "cost_matrix": wdtw_cost_matrix,
        "alignment_path": wdtw_alignment_path,
        "type": DistanceType.ELASTIC,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "wddtw",
        "distance": wddtw_distance,
        "pairwise_distance": wddtw_pairwise_distance,
        "cost_matrix": wddtw_cost_matrix,
        "alignment_path": wddtw_alignment_path,
        "type": DistanceType.ELASTIC,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "lcss",
        "distance": lcss_distance,
        "pairwise_distance": lcss_pairwise_distance,
        "cost_matrix": lcss_cost_matrix,
        "alignment_path": lcss_alignment_path,
        "type": DistanceType.ELASTIC,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "erp",
        "distance": erp_distance,
        "pairwise_distance": erp_pairwise_distance,
        "cost_matrix": erp_cost_matrix,
        "alignment_path": erp_alignment_path,
        "type": DistanceType.ELASTIC,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "edr",
        "distance": edr_distance,
        "pairwise_distance": edr_pairwise_distance,
        "cost_matrix": edr_cost_matrix,
        "alignment_path": edr_alignment_path,
        "type": DistanceType.ELASTIC,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "twe",
        "distance": twe_distance,
        "pairwise_distance": twe_pairwise_distance,
        "cost_matrix": twe_cost_matrix,
        "alignment_path": twe_alignment_path,
        "type": DistanceType.ELASTIC,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "msm",
        "distance": msm_distance,
        "pairwise_distance": msm_pairwise_distance,
        "cost_matrix": msm_cost_matrix,
        "alignment_path": msm_alignment_path,
        "type": DistanceType.ELASTIC,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "adtw",
        "distance": adtw_distance,
        "pairwise_distance": adtw_pairwise_distance,
        "cost_matrix": adtw_cost_matrix,
        "alignment_path": adtw_alignment_path,
        "type": DistanceType.ELASTIC,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "shape_dtw",
        "distance": shape_dtw_distance,
        "pairwise_distance": shape_dtw_pairwise_distance,
        "cost_matrix": shape_dtw_cost_matrix,
        "alignment_path": shape_dtw_alignment_path,
        "type": DistanceType.ELASTIC,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "soft_dtw",
        "distance": soft_dtw_distance,
        "pairwise_distance": soft_dtw_pairwise_distance,
        "cost_matrix": soft_dtw_cost_matrix,
        "alignment_path": soft_dtw_alignment_path,
        "type": DistanceType.ELASTIC,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "sbd",
        "distance": sbd_distance,
        "pairwise_distance": sbd_pairwise_distance,
        "type": DistanceType.CROSS_CORRELATION,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "shift_scale",
        "distance": shift_scale_invariant_distance,
        "pairwise_distance": shift_scale_invariant_pairwise_distance,
        "type": DistanceType.CROSS_CORRELATION,
        "symmetric": False,
        "unequal_support": False,
    },
    {
        "name": "dft_sfa",
        "distance": mindist_dft_sfa_distance,
        "pairwise_distance": mindist_dft_sfa_pairwise_distance,
        "type": DistanceType.MIN_DISTANCE,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "paa_sax",
        "distance": mindist_paa_sax_distance,
        "pairwise_distance": mindist_paa_sax_pairwise_distance,
        "type": DistanceType.MIN_DISTANCE,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "sax",
        "distance": mindist_sax_distance,
        "pairwise_distance": mindist_sax_pairwise_distance,
        "type": DistanceType.MIN_DISTANCE,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "sfa",
        "distance": mindist_sfa_distance,
        "pairwise_distance": mindist_sfa_pairwise_distance,
        "type": DistanceType.MIN_DISTANCE,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "mpdist",
        "distance": mp_distance,
        "pairwise_distance": mp_pairwise_distance,
        "type": DistanceType.MATRIX_PROFILE,
        "symmetric": True,
        "unequal_support": True,
    },
    {
        "name": "swale",
        "distance": swale_distance,
        "pairwise_distance": swale_pairwise_distance,
        "type": DistanceType.ELASTIC,
        "symmetric": True,
        "unequal_support": True,
    },
]

DISTANCES_DICT = {d["name"]: d for d in DISTANCES}
COST_MATRIX = [d["name"] for d in DISTANCES if "cost_matrix" in d]
ALIGNMENT_PATH = [d["name"] for d in DISTANCES if "alignment_path" in d]
PAIRWISE_DISTANCE = [d["name"] for d in DISTANCES if "pairwise_distance" in d]
SYMMETRIC_DISTANCES = [d["name"] for d in DISTANCES if d["symmetric"]]
ASYMMETRIC_DISTANCES = [d["name"] for d in DISTANCES if not d["symmetric"]]
UNEQUAL_LENGTH_SUPPORT_DISTANCES = [
    d["name"] for d in DISTANCES if d["unequal_support"]
]

ELASTIC_DISTANCES = [d["name"] for d in DISTANCES if d["type"] == DistanceType.ELASTIC]
POINTWISE_DISTANCES = [
    d["name"] for d in DISTANCES if d["type"] == DistanceType.POINTWISE
]
MP_DISTANCES = [
    d["name"] for d in DISTANCES if d["type"] == DistanceType.MATRIX_PROFILE
]
MIN_DISTANCES = [d["name"] for d in DISTANCES if d["type"] == DistanceType.MIN_DISTANCE]

# This is a very specific list for testing where a time series of length 1 is not
# supported
SINGLE_POINT_NOT_SUPPORTED_DISTANCES = ["ddtw", "wddtw", "edr"]
