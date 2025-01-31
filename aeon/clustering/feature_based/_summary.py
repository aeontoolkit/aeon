"""Summary Clusterer.

Pipeline clusterer using the basic summary statistics and an estimator.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["SummaryClusterer"]

import numpy as np
from sklearn.cluster import KMeans

from aeon.base._base import _clone_estimator
from aeon.clustering import BaseClusterer
from aeon.transformations.collection.feature_based import SevenNumberSummary
from aeon.utils.tags.enum_tags import AlgorithmType


class SummaryClusterer(BaseClusterer):
    """
    Summary statistic clusterer.

    This clusterer simply transforms the input data using the
    SevenNumberSummary transformer and builds a provided estimator using the
    transformed data.

    Parameters
    ----------
    summary_stats : ["default", "percentiles", "bowley", "tukey"], default="default"
        The summary statistics to compute.
        The options are as follows, with float denoting the percentile value extracted
        from the series:
            - "default": mean, std, min, max, 0.25, 0.5, 0.75
            - "percentiles": 0.215, 0.887, 0.25, 0.5, 0.75, 0.9113, 0.9785
            - "bowley": min, max, 0.1, 0.25, 0.5, 0.75, 0.9
            - "tukey": min, max, 0.125, 0.25, 0.5, 0.75, 0.875
    estimator : sklearn clusterer, default=None
        An sklearn estimator to be built using the transformed data. Defaults to a
        Random Forest with 200 trees.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    See Also
    --------
    SummaryTransformer
    SummaryRegressor

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import KMeans
    >>> from aeon.clustering.feature_based import SummaryClusterer
    >>> X = np.random.random(size=(10,2,20))
    >>> clst = SummaryClusterer(estimator=KMeans(n_clusters=2))
    >>> clst.fit(X)
    SummaryClusterer(...)
    >>> preds = clst.predict(X)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": AlgorithmType.FEATURE.value,
    }

    def __init__(
        self,
        summary_stats="default",
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.summary_stats = summary_stats
        self.estimator = estimator

        self.n_jobs = n_jobs
        self.random_state = random_state

        self._transformer = None
        self._estimator = None

        super().__init__()

    def _fit(self, X, y=None):
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
        self._transformer = SevenNumberSummary(
            summary_stats=self.summary_stats,
        )

        self._estimator = _clone_estimator(
            (KMeans() if self.estimator is None else self.estimator),
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t = self._transformer.fit_transform(X, y)
        self._estimator.fit(X_t, y)

        self.labels_ = self._estimator.labels_

        return self

    def _predict(self, X) -> np.ndarray:
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

    def _predict_proba(self, X) -> np.ndarray:
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
