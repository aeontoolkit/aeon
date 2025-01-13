"""Base class for whole series search."""

__maintainer__ = ["baraline"]

import warnings
from abc import abstractmethod
from typing import Optional, final

import numpy as np
from numba import get_num_threads, set_num_threads

from aeon.similarity_search._base import BaseSimilaritySearch
from aeon.utils.numba.general import compute_mean_stds_collection_parallel


class BaseSeriesSearch(BaseSimilaritySearch):
    """
    Base class for similarity search on whole time series.

    Parameters
    ----------
    normalise : bool, optional
        Whether the inputs should be z-normalised. The default is False.
    n_jobs : int, optional
        Number of parallel jobs to use. The default is 1.
    """

    @final
    def find_motifs(
        self,
        X: np.ndarray,
        k: Optional[int] = 1,
        threshold: Optional[float] = np.inf,
        X_index: Optional[int] = None,
        inverse_distance: Optional[bool] = False,
    ):
        """
        Find the top-k motifs in the training data.

        Given ``k`` and ``threshold`` parameters, this methods returns the top-k motif
        sets. We define a motif set as a set of candidates which all are at a distance
        of at most ``threshold`` from each other. The top-k motifs sets are the
        motif sets with the most candidates.

        Parameters
        ----------
        X : np.ndarray, 2D array of shape (n_channels, n_timestamps)
            A series in which we want to indentify motifs.
        k : int, optional
            Number of motifs to return
        threshold : int, optional
            A threshold on the similarity measure to determine which candidates will be
            part of a motif set.
        X_index : Optional[int], optional
            If ``X`` is a series of the database given in fit, specify its index in
            ``X_``. If specified, this series won't be able to match with itself.
        inverse_distance : bool, optional
            Wheter to inverse the computed distance, meaning that the method will return
            the anomalies instead of motifs.

        Returns
        -------
        ndarray, shape=(k,)
            A numpy array of at most ``k`` elements containing the indexes of the
            motifs in X.
        ndarray, shape=(k,)
            A numpy array of at most ``k`` elements containing the distances of the
            motifs macthes to the motif in X.

        """
        self._check_is_fitted()
        if X is not None:
            self._check_find_neighbors_motif_format(X)
        prev_threads = get_num_threads()
        X_index = self._check_X_index_int(X_index)
        motifs_indexes, distances = self._find_motifs(
            X,
            k=k,
            threshold=threshold,
            inverse_distance=inverse_distance,
            X_index=X_index,
        )
        set_num_threads(prev_threads)
        return motifs_indexes, distances

    @final
    def find_neighbors(
        self,
        X: np.ndarray,
        k: Optional[int] = 1,
        threshold: Optional[float] = np.inf,
        X_index: Optional[int] = None,
        inverse_distance: Optional[bool] = False,
    ):
        """
        Find the top-k neighbors of X in the database.

        Given ``k`` and ``threshold`` parameters, this methods returns the top-k
        neighbors of X, such as each of the ``k`` neighbors as a distance inferior or
        equal to ``threshold``. By default, ``threshold`` is set to infinity. It is
        possible for this method to return less than ``k`` neighbors, either if there
        is less than ``k`` admissible candidate in the database, or if in the top-k
        candidates, some do not meet the ``threshold`` condition.

        Parameters
        ----------
        X : np.ndarray, 2D array of shape (n_channels, length)
            The subsequence for which we want to identify nearest neighbors in the
            database.
        k : int, optional
            Number of neighbors to return.
        threshold : int, optional
            A threshold on the distance to determine which candidates will be returned.
        X_index : Optional[int], optional
            If ``X`` is a series of the database given in fit, specify its index in
            ``X_``. If specified, this series won't be able to match with itself.
        inverse_distance : bool, optional
            Wheter to inverse the computed distance, meaning that the method will return
            the k most dissimilar neighbors instead of the k most similar.


        Returns
        -------
        ndarray, shape=(k,)
            A numpy array of at most ``k`` elements containing the indexes of the
            neighbors.
        ndarray, shape=(k,)
            A numpy array of at most ``k`` elements containing the distances of the
            neighbors to X.

        """
        self._check_is_fitted()

        self._check_find_neighbors_motif_format(X)

        X_index = self._check_X_index_int(X_index)
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        neighbors, distances = self._find_neighbors(
            X,
            k=k,
            threshold=threshold,
            inverse_distance=inverse_distance,
            X_index=X_index,
        )
        set_num_threads(prev_threads)
        if len(neighbors) < k:
            warnings.warn(
                f"The number of admissible neighbors found is {len(neighbors)}, instead"
                f" of {k}",
                stacklevel=2,
            )
        return neighbors, distances

    def _compute_mean_std_from_collection(self, X: np.ndarray):
        """
        Compute the mean and std of each channel for all series in X.

        Parameters
        ----------
        X : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
            Collection of series from which we extract mean and stds. If it is an
            unequal length collection, it should be a list of 2d numpy arrays.

        Returns
        -------
        Tuple(np.ndarray, np.ndarray)
            Both array are of shape (n_cases, n_channels), the first contains the means
            and the second the stds for each series in X.

        """
        means, stds = compute_mean_stds_collection_parallel(X)
        return means, stds

    def _fit(self, X, y=None):
        if self.normalise:
            self.X_means_, self.X_stds_ = self._compute_mean_std_from_collection(X)
        self.X_ = X
        return self

    @abstractmethod
    def _find_motifs(
        self,
        X: np.ndarray,
        k: Optional[int] = 1,
        threshold: Optional[float] = np.inf,
        X_index: Optional[int] = None,
        inverse_distance: Optional[bool] = False,
    ): ...

    @abstractmethod
    def _find_neighbors(
        self,
        X: np.ndarray,
        k: Optional[int] = 1,
        threshold: Optional[float] = np.inf,
        inverse_distance: Optional[bool] = False,
        X_index=None,
    ): ...


# TODO : Add an update method to add series to the index
class BaseIndexSearch(BaseSeriesSearch):
    """
    Base class for similarity search on whole time series using indexes.

    Parameters
    ----------
    normalise : bool, optional
        Whether the inputs should be z-normalised. The default is False.
    n_jobs : int, optional
        Number of parallel jobs to use. The default is 1.
    """

    def _fit(self, X, y=None):
        super()._fit(X)
        self._build_index()
        return self

    @abstractmethod
    def _build_index(self): ...

    @abstractmethod
    def _query_index(
        self,
        X,
        k=1,
        inverse_distance=False,
        threshold=np.inf,
    ): ...

    @abstractmethod
    def _get_bucket_sizes(self): ...

    @abstractmethod
    def _get_bucket_content(self, key): ...

    def _find_motifs(
        self,
        X: np.ndarray,
        k: Optional[int] = 1,
        threshold: Optional[float] = np.inf,
        X_index: Optional[int] = None,
        inverse_distance: Optional[bool] = False,
    ):
        bucket_sizes = self._get_bucket_sizes()
        idx_motifs = np.argsort(list(bucket_sizes.values()))[::-1][:, k]
        # TODO : review distance return on motif for whole series and buckets
        return [self._get_bucket_content(idx_motif) for idx_motif in idx_motifs], [
            0 for _ in idx_motifs
        ]

    def _find_neighbors(
        self,
        X: np.ndarray,
        k: Optional[int] = 1,
        threshold: Optional[float] = np.inf,
        inverse_distance: Optional[bool] = False,
        X_index=None,
    ):
        top_k, top_k_dist = self._query_index(
            X, k=k, inverse_distance=inverse_distance, threshold=threshold
        )
        return top_k, top_k_dist
