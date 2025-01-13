"""CBLOF for Anomaly Detection."""

__maintainer__ = []
__all__ = ["CBLOF"]

from typing import Optional, Union

import numpy as np

from aeon.anomaly_detection._pyodadapter import PyODAdapter
from aeon.utils.validation._dependencies import _check_soft_dependencies


class CBLOF(PyODAdapter):
    r"""CBLOF for Anomaly Detection.

    This class implements the CBLOF algorithm for anomaly detection
    using PyODAdadpter to be used in the aeon framework. All parameters are passed to
    the PyOD model ``CBLOF`` except for `window_size` and `stride`, which are used to
    construct the sliding windows.

    The documentation for parameters has been adapted from the
    [PyOD documentation](https://pyod.readthedocs.io/en/latest/pyod.models.html#id117).
    Here, `X` refers to the set of sliding windows extracted from the time series
    using :func:`aeon.utils.windowing.sliding_windows` with the parameters
    ``window_size`` and ``stride``. The internal `X` has the shape
    `(n_windows, window_size * n_channels)`.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    clustering_estimator : Estimator or None, default=None
        The base clustering algorithm for performing data clustering.
        A valid clustering algorithm should be passed in. The estimator should
        have standard sklearn APIs, fit() and predict(). The estimator should
        have attributes ``labels_`` and ``cluster_centers_``.
        If ``cluster_centers_`` is not in the attributes once the model is fit,
        it is calculated as the mean of the samples in a cluster.

        If not set, CBLOF uses KMeans for scalability. See
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

        aeon clustering estimators are not supported.

    alpha : float in (0.5, 1), default=0.9
        Coefficient for deciding small and large clusters. The ratio
        of the number of samples in large clusters to the number of samples in
        small clusters.

    beta : int or float in (1,), default=5
        Coefficient for deciding small and large clusters. For a list
        sorted clusters by size `|C1|, \|C2|, ..., |Cn|, beta = |Ck|/|Ck-1|`

    use_weights : bool, default=False
        If set to True, the size of clusters are used as weights in
        outlier score calculation.

    check_estimator : bool, default=False
        If set to True, check whether the base estimator is consistent with
        sklearn standard.

    random_state : int, np.RandomState or None, default=None
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    window_size : int, default=10
        Size of the sliding window.

    stride : int, default=1
        Stride of the sliding window.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:univariate": True,
        "capability:missing_values": False,
        "fit_is_empty": False,
        "python_dependencies": ["pyod"],
    }

    def __init__(
        self,
        n_clusters: int = 8,
        clustering_estimator=None,
        alpha: float = 0.9,
        beta: Union[int, float] = 5,
        use_weights: bool = False,
        check_estimator: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        window_size: int = 10,
        stride: int = 1,
    ):
        _check_soft_dependencies(*self._tags["python_dependencies"])
        from pyod.models.cblof import CBLOF

        model = CBLOF(
            n_clusters=n_clusters,
            clustering_estimator=clustering_estimator,
            alpha=alpha,
            beta=beta,
            use_weights=use_weights,
            check_estimator=check_estimator,
            random_state=random_state,
        )
        self.n_clusters = n_clusters
        self.clustering_estimator = clustering_estimator
        self.alpha = alpha
        self.beta = beta
        self.use_weights = use_weights
        self.check_estimator = check_estimator
        self.random_state = random_state
        super().__init__(model, window_size, stride)

    def _fit(self, X: np.ndarray, y: Union[np.ndarray, None] = None) -> None:
        super()._fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return super()._predict(X)

    def _fit_predict(
        self, X: np.ndarray, y: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        return super()._fit_predict(X, y)

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
        params : dict
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return {
            "n_clusters": 4,
            "alpha": 0.75,
            "beta": 3,
        }
