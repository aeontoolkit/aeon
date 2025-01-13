"""Mock series transformers useful for testing and debugging."""

__maintainer__ = []
__all__ = [
    "MockSubsequenceSearch",
    "MockMatrixProfile",
]

import numpy as np

from aeon.similarity_search.subsequence_search._base import (
    BaseMatrixProfile,
    BaseSubsequenceSearch,
)


class MockMatrixProfile(BaseMatrixProfile):
    """Mock estimator for BaseMatrixProfile."""

    def __init__(
        self,
        length=3,
        normalise=False,
        n_jobs=1,
    ):
        super().__init__(length=length, n_jobs=n_jobs, normalise=normalise)

    def compute_matrix_profile(
        self,
        k,
        threshold,
        exclusion_size,
        inverse_distance,
        allow_neighboring_matches,
        X=None,
        X_index=None,
    ):
        """Compute matrix profiles between X_ and X or between all series in X_."""
        return np.zeros((X.shape[1] - self.length + 1, k)), np.zeros(
            (X.shape[1] - self.length + 1, k, 2), dtype=np.int64
        )

    def compute_distance_profile(self, X):
        """Compute distrance profiles between X_ and X (a series of size length)."""
        return [
            np.zeros(self.X_[i].shape[1] - self.length + 1) for i in range(len(self.X_))
        ]


class MockSubsequenceSearch(BaseSubsequenceSearch):
    """Mock estimator for BaseSubsequenceSearch."""

    def __init__(
        self,
        length=3,
        normalise=False,
        n_jobs=1,
    ):
        super().__init__(length=length, n_jobs=n_jobs, normalise=normalise)

    def _find_motifs(
        self,
        X,
        k=1,
        threshold=np.inf,
        X_index=None,
        inverse_distance=False,
        allow_neighboring_matches=False,
        exclusion_factor=2.0,
    ):
        return [[0, 0]], self.X_[0][0:1]  # TODO: update after logic is implemented

    def _find_neighbors(
        self,
        X,
        k=1,
        threshold=np.inf,
        inverse_distance=False,
        X_index=None,
        allow_neighboring_matches=False,
        exclusion_factor=2.0,
    ):
        return [[0, 0]], self.X_[0][0:1]
