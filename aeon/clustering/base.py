"""Base class for clustering."""

__maintainer__ = []
__all__ = ["BaseClusterer"]

from abc import abstractmethod
from typing import final

import numpy as np
from sklearn.base import ClusterMixin

from aeon.base import BaseCollectionEstimator


class BaseClusterer(ClusterMixin, BaseCollectionEstimator):
    """Abstract base class for time series clusterers.

    Parameters
    ----------
    n_clusters : int, default=None
        Number of clusters for model.
    """

    _tags = {
        "fit_is_empty": False,
    }

    @abstractmethod
    def __init__(self):
        super().__init__()

    @final
    def fit(self, X, y=None) -> BaseCollectionEstimator:
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X : 3D np.ndarray (any number of channels, equal length series)
                of shape (n_cases, n_channels, n_timepoints)
            or 2D np.array (univariate, equal length series)
                of shape (n_cases, n_timepoints)
            or list of numpy arrays (any number of channels, unequal length series)
                of shape [n_cases], 2D np.array (n_channels, n_timepoints_i), where
                n_timepoints_i is length of series i
            other types are allowed and converted into one of the above.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        self:
            Fitted estimator.
        """
        self.reset()
        X = self._preprocess_collection(X)
        self._fit(X)
        self.is_fitted = True
        return self

    @final
    def predict(self, X) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. Other types are
            allowed and converted into one of the above.

        Returns
        -------
        np.array
            shape ``(n_cases)`, index of the cluster each time series in X.
            belongs to.
        """
        self._check_is_fitted()
        X = self._preprocess_collection(X, store_metadata=False)
        self._check_shape(X)
        return self._predict(X)

    @final
    def predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Default behaviour is to call _predict and set the predicted class probability
        to 1, other class probabilities to 0. Override if better estimates are
        obtainable.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. Other types are
            allowed and converted into one of the above.

        Returns
        -------
        y : 2D array of shape [n_cases, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        self._check_is_fitted()
        X = self._preprocess_collection(X, store_metadata=False)
        self._check_shape(X)
        return self._predict_proba(X)

    @final
    def fit_predict(self, X, y=None) -> np.ndarray:
        """Compute cluster centers and predict cluster index for each time series.

        Convenience method; equivalent of calling fit(X) followed by predict(X)

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_cases, n_timepoints) or shape
            (n_cases, n_channels, n_timepoints)).
            Time series instances to train clusterer and then have indexes each belong
            to return.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_cases,))
            Index of the cluster each time series in X belongs to.
        """
        self.fit(X)
        return self.labels_

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Default behaviour is to call _predict and set the predicted class probability
        to 1, other class probabilities to 0. Override if better estimates are
        obtainable.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. Other types are
            allowed and converted into one of the above.

        Returns
        -------
        y : 2D array of shape [n_cases, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        preds = self._predict(X)
        unique = np.unique(preds)
        for i, u in enumerate(unique):
            preds[preds == u] = i
        n_cases = len(preds)
        if hasattr(self, "n_clusters"):
            n_clusters = self.n_clusters
        else:
            n_clusters = len(np.unique(preds))
        if n_clusters is None:
            n_clusters = int(max(preds)) + 1
        dists = np.zeros((len(X), n_clusters))
        for i in range(n_cases):
            dists[i, preds[i]] = 1
        return dists

    @abstractmethod
    def _predict(self, X) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_cases, n_timepoints) or shape
            (n_cases,n_channels,n_timepoints)).
            Time series instances to predict their cluster indexes.

        Returns
        -------
        np.ndarray (1d array of shape (n_cases,))
            Index of the cluster each time series in X belongs to.
        """
        ...

    @abstractmethod
    def _fit(self, X, y=None):
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_cases, n_timepoints) or shape
            (n_cases,n_channels,n_timepoints)).
            Training time series instances to cluster.

        Returns
        -------
        self:
            Fitted estimator.
        """
        ...
