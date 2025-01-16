"""Base similiarity search for series."""

from abc import abstractmethod
from typing import Union, final

import numpy as np
from numba import get_num_threads, set_num_threads

from aeon.base import BaseSeriesEstimator
from aeon.similarity_search._base import BaseSimilaritySearch
from aeon.utils.validation import check_n_jobs


class BaseSeriesSimilaritySearch(BaseSeriesEstimator, BaseSimilaritySearch):
    """Base class for similarity search applications on single series."""

    _tags = {
        "input_data_type": "Series",
    }

    @abstractmethod
    def __init__(self, axis=1, n_jobs=1):
        self.n_jobs = n_jobs
        super().__init__(axis=axis)

    @final
    def fit(
        self,
        X: np.ndarray,
        y=None,
    ):
        """
        Fit method: data preprocessing and storage.

        Parameters
        ----------
        X : np.ndarray, 2D array of shape (n_channels, n_timepoints)
            Input series to be used for the similarity search operations.
        y : optional
            Not used.

        Raises
        ------
        TypeError
            If the input X array is not 2D raise an error.

        Returns
        -------
        self
        """
        self.reset()
        self._n_jobs = check_n_jobs(self.n_jobs)
        X = self._preprocess_series(X, self.axis, True)
        # Store minimum number of n_timepoints for unequal length collections
        self.n_channels_ = X.shape[0]
        self.n_timepoints_ = X.shape[1]
        self.X_ = X

        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        self._fit(X, y=y)
        set_num_threads(prev_threads)

        self.is_fitted = True
        return self

    @abstractmethod
    def _fit(
        self,
        X: np.ndarray,
        y=None,
    ): ...

    def _pre_predict(
        self,
        X: Union[np.ndarray, None] = None,
    ):
        """
        Predict method.

        Parameters
        ----------
        X : Union[np.ndarray, None], optional
            Optional data to use for predict.. The default is None.

        """
        self._check_is_fitted()
        if X is not None:
            X = self._preprocess_series(X, self.axis, False)
            self._check_predict_format(X)
        return X

    def _check_X_index(self, X_index: int):
        """
        Check wheter a X_index parameter is correctly formated and is admissible.

        Parameters
        ----------
        X_index : int
            Index of a timestamp in X_.

        """
        if X_index is not None:
            if not isinstance(X_index, int):
                raise TypeError("Expected an integer for X_index but got {X_index}")

            max_timepoints = self.n_timepoints_
            if hasattr(self, "length"):
                max_timepoints -= self.length
            if X_index >= max_timepoints or X_index < 0:
                raise ValueError(
                    "The value of X_index cannot exced the number "
                    "of timepoint in series given during fit. Expected a value "
                    f"between [0, {max_timepoints - 1}] but got {X_index}"
                )
