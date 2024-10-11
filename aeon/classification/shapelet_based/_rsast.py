"""Random Scalable and Accurate Subsequence Transform (RSAST).

Pipeline classifier using the RSAST transformer and an sklearn classifier.
"""

__maintainer__ = ["nirojasva"]
__all__ = ["RSASTClassifier"]

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from aeon.transformations.collection.shapelet_based import RSAST


class RSASTClassifier(BaseClassifier):
    """RSASTClassifier.

    Classification pipeline using
    Random Scalable and Accurate Subsequence Transform (RSAST) [1]_ transformer
    and an sklearn classifier.

    Parameters
    ----------
    n_random_points: int default = 10 the number of initial random points to extract
    len_method:  string default="both" the type of statistical tool used to get the
    length of shapelets. "both"=ACF&PACF, "ACF"=ACF, "PACF"=PACF,
    "None"=Extract randomly any length from the TS
    nb_inst_per_class : int default = 10
        the number of reference time series to select per class
    seed : int, default = None
        the seed of the random generator
    classifier : sklearn compatible classifier, default = None
        if None, a RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)) is used.
    class_weight{“balanced”, “balanced_subsample”}: dict or list of dicts, default=None
        From sklearn documentation:
        If not given, all classes are supposed to have weight one.
        The “balanced” mode uses the values of y to automatically adjust weights
        inversely proportional to class frequencies in the input data as
        n_samples / (n_classes * np.bincount(y))
        The “balanced_subsample” mode is the same as “balanced” except that weights
        are computed based on the bootstrap sample for every tree grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed through
        the fit method) if sample_weight is specified.
    n_jobs : int, default -1
        Number of threads to use for the transform.


    References
    ----------
    .. [1] Varela, N. R., Mbouopda, M. F., & Nguifo, E. M. (2023). RSAST: Sampling
    Shapelets for Time Series Classification.
    https://hal.science/hal-04311309/

    Examples
    --------
    >>> from aeon.classification.shapelet_based import RSASTClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = RSASTClassifier() # doctest: +SKIP
    >>> clf.fit(X_train, y_train) # doctest: +SKIP
    RSASTClassifier(...)
    >>> y_pred = clf.predict(X_test) # doctest: +SKIP
    """

    _tags = {
        "capability:multithreading": True,
        "capability:multivariate": False,
        "algorithm_type": "shapelet",
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        n_random_points=10,
        len_method="both",
        nb_inst_per_class=10,
        seed=None,
        classifier=None,
        class_weight=None,
        n_jobs=-1,
    ):
        super().__init__()
        self.n_random_points = n_random_points
        self.len_method = len_method
        self.nb_inst_per_class = nb_inst_per_class
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.seed = seed
        self.classifier = classifier

    def _fit(self, X, y):
        """Fit RSASTClassifier to the training data.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The training input samples.
        y: array-like or list
            The class values for X.

        Returns
        -------
        self : RSASTClassifier
            This pipeline classifier

        """
        self._transformer = RSAST(
            self.n_random_points,
            self.len_method,
            self.nb_inst_per_class,
            self.seed,
            self.class_weight,
            self.n_jobs,
        )

        self._classifier = _clone_estimator(
            (
                RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight=self.class_weight)
                if self.classifier is None
                else self.classifier
            ),
            self.seed,
        )

        self._pipeline = make_pipeline(self._transformer, self._classifier)

        self._pipeline.fit(X, y)

        return self

    def _predict(self, X):
        """Predict labels for the input.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The training input samples.

        Returns
        -------
        array-like or list
            Predicted class labels.
        """
        return self._pipeline.predict(X)

    def _predict_proba(self, X):
        """Predict labels probabilities for the input.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The training input samples.

        Returns
        -------
        dists : np.ndarray shape (n_cases, n_timepoints)
            Predicted class probabilities.
        """
        m = getattr(self._classifier, "predict_proba", None)
        if callable(m):
            dists = self._pipeline.predict_proba(X)
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._pipeline.predict(X)
            for i in range(0, X.shape[0]):
                dists[i, np.where(self.classes_ == preds[i])] = 1
        return dists
