"""Abstract base class for whole-series/collection anomaly detectors."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["BaseCollectionAnomalyDetector"]

from abc import abstractmethod
from typing import final

import numpy as np
import pandas as pd

from aeon.base import BaseCollectionEstimator


class BaseCollectionAnomalyDetector(BaseCollectionEstimator):
    """Collection anomaly detector base class."""

    _tags = {
        "fit_is_empty": False,
        "requires_y": False,
    }

    def __init__(self):
        super().__init__()

    @final
    def fit(self, X, y=None):
        """Fit."""
        if self.get_tag("fit_is_empty"):
            self.is_fitted = True
            return self

        if self.get_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")

        # reset estimator at the start of fit
        self.reset()

        X = self._preprocess_collection(X)
        if y is not None:
            y = self._check_y(y, self.metadata_["n_cases"])

        self._fit(X, y)

        # this should happen last
        self.is_fitted = True
        return self

    @final
    def predict(self, X):
        """Predict."""
        fit_empty = self.get_tag("fit_is_empty")
        if not fit_empty:
            self._check_is_fitted()

        X = self._preprocess_collection(X, store_metadata=False)
        # Check if X has the correct shape seen during fitting
        self._check_shape(X)

        return self._predict(X)

    @abstractmethod
    def _fit(self, X, y=None): ...

    @abstractmethod
    def _predict(self, X): ...

    def _check_y(self, y, n_cases):
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError(
                f"y must be a np.array or a pd.Series, but found type: {type(y)}"
            )
        if isinstance(y, np.ndarray) and y.ndim > 1:
            raise TypeError(f"y must be 1-dimensional, found {y.ndim} dimensions")

        if not all([x == 0 or x == 1 for x in y]):
            raise ValueError(
                "y input must only contain 0 (not anomalous) or 1 (anomalous) values."
            )

        # Check matching number of labels
        n_labels = y.shape[0]
        if n_cases != n_labels:
            raise ValueError(
                f"Mismatch in number of cases. Found X = {n_cases} and y = {n_labels}"
            )

        if isinstance(y, pd.Series):
            y = pd.Series.to_numpy(y)

        return y
