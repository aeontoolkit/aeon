"""Scalable and Accurate Subsequence Transform (SAST).

Pipeline classifier using the SAST transformer and an sklearn classifier.
"""

from typing import Optional

__maintainer__ = ["TonyBagnall"]
__all__ = ["SASTClassifier"]

from operator import itemgetter

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from aeon.transformations.collection.shapelet_based import SAST
from aeon.utils.numba.general import z_normalise_series


class SASTClassifier(BaseClassifier):
    """Classification pipeline using SAST [1]_ transformer and an sklean classifier.

    Parameters
    ----------
    length_list : int[], default = None
        an array containing the lengths of the subsequences to be generated.
        If None, will be infered during fit as np.arange(3, X.shape[1])
    stride : int, default = 1
        the stride used when generating subsquences
    nb_inst_per_class : int default = 1
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
    .. [1] Mbouopda, Michael Franklin, and Engelbert Mephu Nguifo.
           "Scalable and accurate subsequence transform for time series classification."
           Pattern Recognition 147 (2023): 110121.
           https://www.sciencedirect.com/science/article/abs/pii/S003132032300818X,
           https://uca.hal.science/hal-03087686/document

    Examples
    --------
    >>> from aeon.classification.shapelet_based import SASTClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = SASTClassifier()
    >>> clf.fit(X_train, y_train)
    SASTClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multithreading": True,
        "capability:multivariate": False,
        "algorithm_type": "shapelet",
    }

    def __init__(
        self,
        length_list=None,
        stride: int = 1,
        nb_inst_per_class: int = 1,
        seed: Optional[int] = None,
        classifier=None,
        class_weight=None,
        n_jobs: int = -1,
    ) -> None:
        super().__init__()
        self.length_list = length_list
        self.stride = stride
        self.nb_inst_per_class = nb_inst_per_class
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.seed = seed

        self.classifier = classifier

    def _fit(self, X, y):
        """Fit SASTClassifier to the training data.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The training input samples.
        y: array-like or list
            The class values for X.

        Returns
        -------
        self : SASTClassifier
            This pipeline classifier

        """
        self._transformer = SAST(
            self.length_list,
            self.stride,
            self.nb_inst_per_class,
            self.seed,
            self.class_weight,
            self.n_jobs,
        )

        self._classifier = _clone_estimator(
            (
                RidgeClassifierCV(
                    alphas=np.logspace(-3, 3, 10), class_weight=self.class_weight
                    )
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

    def plot_most_important_feature_on_ts(self, ts, feature_importance, limit: int = 5):
        """Plot the most important features on ts.

        Parameters
        ----------
        ts : float[:]
            The time series
        feature_importance : float[:]
            The importance of each feature in the transformed data
        limit : int, default = 5
            The maximum number of features to plot

        Returns
        -------
        fig : plt.figure
            The figure
        """
        import matplotlib.pyplot as plt

        features = zip(self._transformer._kernel_orig, feature_importance)
        sorted_features = sorted(features, key=itemgetter(1), reverse=True)

        max_ = min(limit, len(sorted_features))

        fig, axes = plt.subplots(
            1, max_, sharey=True, figsize=(3 * max_, 3), tight_layout=True
        )

        for f in range(max_):
            kernel, _ = sorted_features[f]
            znorm_kernel = z_normalise_series(kernel)
            d_best = np.inf
            for i in range(ts.size - kernel.size):
                s = ts[i : i + kernel.size]
                s = z_normalise_series(s)
                d = np.sum((s - znorm_kernel) ** 2)
                if d < d_best:
                    d_best = d
                    start_pos = i
            axes[f].plot(range(start_pos, start_pos + kernel.size), kernel, linewidth=5)
            axes[f].plot(range(ts.size), ts, linewidth=2)
            axes[f].set_title(f"feature: {f+1}")

        return fig
