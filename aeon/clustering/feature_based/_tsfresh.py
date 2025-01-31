"""TSFresh Clusterer.

Pipeline clusterer using the TSFresh transformer and an estimator.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["TSFreshClusterer"]


from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from aeon.base._base import _clone_estimator
from aeon.clustering import BaseClusterer
from aeon.transformations.collection.feature_based import TSFresh
from aeon.utils.tags.enum_tags import AlgorithmType


class TSFreshClusterer(BaseClusterer):
    """
    Time Series Feature Extraction based on Scalable Hypothesis Tests clusterer.

    This clusterer simply transforms the input data using the TSFresh [1]_
    transformer and builds a provided estimator using the transformed data.

    Parameters
    ----------
    default_fc_parameters : str, default="efficient"
        Set of TSFresh features to be extracted, options are "minimal", "efficient" or
        "comprehensive".
    estimator : sklearn clusterer, default=None
        An sklearn estimator to be built using the transformed data. Defaults to a
        Random Forest with 200 trees.
    verbose : int, default=0
        Level of output printed to the console (for information only).
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    chunksize : int or None, default=None
        Number of series processed in each parallel TSFresh job, should be optimised
        for efficient parallelisation.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_clusters : int, default=8
        Number of clusters for KMeans (or other estimators that support n_clusters).

    See Also
    --------
    TSFresh

    References
    ----------
    .. [1] Christ, Maximilian, et al. "Time series feature extraction on basis of
        scalable hypothesis tests (tsfresh–a python package)." Neurocomputing 307
        (2018): 72-77.
        https://www.sciencedirect.com/science/article/pii/S0925231218304843

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import KMeans
    >>> from aeon.clustering.feature_based import TSFreshClusterer
    >>> X = np.random.random(size=(10,2,20))
    >>> clst = TSFreshClusterer(estimator=KMeans(n_clusters=2))  # doctest: +SKIP
    >>> clst.fit(X)  # doctest: +SKIP
    TSFreshClusterer(...)
    >>> preds = clst.predict(X)  # doctest: +SKIP
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": AlgorithmType.FEATURE.value,
        "python_dependencies": "tsfresh",
    }

    def __init__(
        self,
        default_fc_parameters: str = "efficient",
        estimator=None,
        verbose: int = 0,
        n_jobs: int = 1,
        chunksize: Optional[int] = None,
        random_state: Optional[int] = None,
        n_clusters: int = 8,  # Default value as 8
    ):
        self.default_fc_parameters = default_fc_parameters
        self.estimator = estimator

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.random_state = random_state
        self.n_clusters = n_clusters

        self._transformer = None
        self._estimator = None

        super().__init__()

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit a pipeline on cases X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The training data.
        y : array-like, shape = [n_cases]
            Ignored. The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        self._transformer = TSFresh(
            default_fc_parameters=self.default_fc_parameters,
            n_jobs=self._n_jobs,
            chunksize=self.chunksize,
        )

        n_clusters = 8 if self.n_clusters is None else self.n_clusters

        if self.estimator is None:
            self._estimator = _clone_estimator(
                KMeans(n_clusters=n_clusters), self.random_state
            )
        else:
            if (
                hasattr(self.estimator, "n_clusters")
                and self.estimator.n_clusters is None
            ):
                self.estimator.n_clusters = self.n_clusters

            self._estimator = _clone_estimator(self.estimator, self.random_state)

        if self.verbose < 2:
            self._transformer.show_warnings = False
            if self.verbose < 1:
                self._transformer.disable_progressbar = True

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t = self._transformer.fit_transform(X, y)

        if X_t.shape[1] == 0:
            raise RuntimeError(
                "No features extracted, try changing the default_fc_parameters to "
                "include more features or disable the relevant feature extractor."
            )
        else:
            self._estimator.fit(X_t, y)

        self.labels_ = self._estimator.labels_
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted class labels.
        """
        return self._estimator.predict(self._transformer.transform(X))

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predictions for.

        Returns
        -------
        y : 2D array of shape [n_cases, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(self._transformer.transform(X))
        else:
            return super()._predict_proba(X)

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
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
        return {
            "default_fc_parameters": "minimal",
            "n_clusters": 3,
        }
