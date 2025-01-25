"""Base class for similarity search."""

__maintainer__ = ["baraline"]
__all__ = [
    "BaseSimilaritySearch",
]


from abc import abstractmethod
from typing import Union

import numpy as np
from numba.typed import List

from aeon.base import BaseAeonEstimator


class BaseSimilaritySearch(BaseAeonEstimator):
    """Base class for similarity search applications."""

    _tags = {
        "requires_y": False,
        "fit_is_empty": False,
    }

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, List],
        y=None,
    ):
        """
        Fit estimator to X.

        State change:
            Changes state to "fitted".

        Writes to self:
        _is_fitted : flag is set to True.

        Parameters
        ----------
        X : Series or Collection, any supported type
            Data to fit transform to, of python type as follows:
                Series: 2D np.ndarray shape (n_channels, n_timepoints)
                Collection: 3D np.ndarray shape (n_cases, n_channels, n_timepoints)
                or list of 2D np.ndarray, case i has shape (n_channels, n_timepoints_i)
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        self : a fitted instance of the estimator
        """
        ...

    @abstractmethod
    def predict(
        self,
        X: Union[np.ndarray, None] = None,
    ):
        """
        Predict method.

        Parameters
        ----------
        X : 2D np.array of shape ``(n_cases, n_timepoints)``
            Optional data to use for predict.
        """
        ...

    def _check_predict_series_format(self, X, length=None):
        """
        Check wheter a series X in predict is correctly formated.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_timepoints)
            A series to be used in predict.
        """
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise TypeError(
                    "A np.ndarray given in predict must be 2D"
                    f"(n_channels, n_timepoints) but found {X.ndim}D."
                )
        else:
            raise TypeError(
                "Expected a 2D np.ndarray in predict but found" f" {type(X)}."
            )
        if self.n_channels_ != X.shape[0]:
            raise ValueError(
                f"Expected X to have {self.n_channels_} channels but"
                f" got {X.shape[0]} channels."
            )
        if length is not None and X.shape[1] != length:
            raise ValueError(
                f"Expected X to have {length} timepoints but"
                f" got {X.shape[1]} timepoints."
            )
