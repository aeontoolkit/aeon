"""Scalable and Accurate Subsequence Transform (SAST).

Pipeline classifier using the SAST transformer and an sklearn classifier.
"""

__author__ = ["MichaelMbouopda"]
__all__ = ["SASTClassifier"]

from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from aeon.transformations.collection.subsequence_based import SAST


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
    random_state : int, default = None
        the seed of the random generator
    classifier : sklearn compatible classifier, default = None
        if None, a RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)) is used.
    n_jobs : int, default -1
        Number of threads to use for the transform.


    Reference
    ---------
    .. [1] Mbouopda, Michael Franklin, and Engelbert Mephu Nguifo.
    "Scalable and accurate subsequence transform for time series classification."
    Pattern Recognition 147 (2023): 110121.
    https://www.sciencedirect.com/science/article/abs/pii/S003132032300818X,
    https://uca.hal.science/hal-03087686/document

    Examples
    --------
    >>> from aeon.classification.subsequence_based import SASTClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = SASTClassifier(num_kernels=500)
    >>> clf.fit(X_train, y_train)
    SASTClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multithreading": True,
        "capability:multivariate": False,
        "algorithm_type": "subsequence",
    }

    def __init__(
        self,
        length_list=None,
        stride=1,
        nb_inst_per_class=1,
        random_state=None,
        classifier=None,
        n_jobs=-1,
    ):
        super(SASTClassifier, self).__init__()
        self.length_list = length_list
        self.stride = stride
        self.nb_inst_per_class = nb_inst_per_class
        self.n_jobs = n_jobs
        self.random_state = (
            np.random.RandomState(random_state)
            if not isinstance(random_state, np.random.RandomState)
            else random_state
        )

        self.classifier = classifier

    def _fit(self, X, y):
        """Fit SASTClassifier to the training data.

        Parameters
        ----------
        X : float[:,:,:]
            an array of shape (n_time_series, n_channels,
            time_series_length) containing the time series
        y : Any[:]
            an array of shape (n_time_series,), containing
            the class label of each time series in X

        Return
        ------
        self

        """
        self._transformer = SAST(
            self.length_list,
            self.stride,
            self.nb_inst_per_class,
            self.random_state,
            self.n_jobs,
        )

        self._classifier = _clone_estimator(
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            if self.classifier is None
            else self.classifier,
            self.random_state,
        )

        self.pipeline = make_pipeline(self._transformer, self._classifier)

        self.pipeline.fit(X, y)

        return self

    def _predict(self, X):
        """Predict labels for the input.

        Parameters
        ----------
        X : float[:,:,:]
            an array of shape (n_time_series, n_channels,
            time_series_length) containing the time series

        Return
        ------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        return self.pipeline.predict(X)

    def _predict_proba(self, X):
        """Predict labels probabilities for the input.

        Parameters
        ----------
        X : float[:,:,:]
            an array of shape (n_time_series, n_channels,
            time_series_length) containing the time series

        Return
        ------
        y : array-like, shape = [n_instances, n_classes]
            Predicted class probabilities.
        """
        return self.pipeline_.predict_proba(X)

    def plot_most_important_feature_on_ts(self, ts, feature_importance, limit=5):
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
        plt figure
        """
        features = zip(self._transformer.kernel_orig_, feature_importance)
        sorted_features = sorted(features, key=itemgetter(1), reverse=True)

        max_ = min(limit, len(sorted_features))

        fig, axes = plt.subplots(
            1, max_, sharey=True, figsize=(3 * max_, 3), tight_layout=True
        )

        for f in range(max_):
            kernel, _ = sorted_features[f]
            znorm_kernel = (kernel - kernel.mean()) / (kernel.std() + 1e-8)
            d_best = np.inf
            for i in range(ts.size - kernel.size):
                s = ts[i : i + kernel.size]
                s = (s - s.mean()) / (s.std() + 1e-8)
                d = np.sum((s - znorm_kernel) ** 2)
                if d < d_best:
                    d_best = d
                    start_pos = i
            axes[f].plot(range(start_pos, start_pos + kernel.size), kernel, linewidth=5)
            axes[f].plot(range(ts.size), ts, linewidth=2)
            axes[f].set_title(f"feature: {f+1}")

        return fig
