"""Base class for subsequence search."""

__maintainer__ = ["baraline"]

import warnings
from abc import abstractmethod
from typing import Optional, final

import numpy as np
from numba import get_num_threads, set_num_threads
from numba.typed import List

from aeon.similarity_search._base import BaseSimilaritySearch
from aeon.similarity_search.subsequence_search._commons import (
    _extract_top_k_from_dist_profile,
    _inverse_distance_profile_list,
)
from aeon.utils.numba.general import sliding_mean_std_one_series

# We can define a BaseVariableLengthSubsequenceSearch later for VALMOD and the likes.

# BaseSubSeries 'replace sub by series'


class BaseSubsequenceSearch(BaseSimilaritySearch):
    """
    Base class for similarity search on time series subsequences.

    Parameters
    ----------
    length : int
        The length of the subsequence to be considered.
    normalise : bool, optional
        Whether the inputs should be z-normalised. The default is False.
    n_jobs : int, optional
        Number of parallel jobs to use. The default is 1.
    """

    @abstractmethod
    def __init__(
        self,
        length: int,
        normalise: Optional[bool] = False,
        n_jobs: Optional[int] = 1,
    ):
        self.length = length
        super().__init__(n_jobs=n_jobs, normalise=normalise)

    _tags = {
        "capability:unequal_length": True,
    }

    @final
    def find_motifs(
        self,
        X: np.ndarray,
        k: Optional[int] = 1,
        threshold: Optional[float] = np.inf,
        X_index: Optional[int] = None,
        inverse_distance: Optional[bool] = False,
        allow_neighboring_matches: Optional[bool] = False,
        exclusion_factor: Optional[float] = 2.0,
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
            ``X_``. If specified, each query of this series won't be able to match with
            its neighboring subsequences.
        inverse_distance : bool, optional
            Wheter to inverse the computed distance, meaning that the method will return
            the anomalies instead of motifs.
        allow_neighboring_matches: bool, optional
            Wheter a candidate can be part of multiple motif sets (True), or if motif
            sets should be mutually exclusive (False).
        exclusion_factor : float, default=2.
            A factor of the query length used to define the exclusion zone when
            ``allow_neighboring_matches`` is set to False. For a given timestamp,
            the exclusion zone starts from
            :math:`id_timestamp - query_length//exclusion_factor` and end at
            :math:`id_timestamp + query_length//exclusion_factor`.

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
            exclusion_factor=exclusion_factor,
            inverse_distance=inverse_distance,
            allow_neighboring_matches=allow_neighboring_matches,
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
        inverse_distance: Optional[bool] = False,
        X_index: Optional[np.ndarray] = None,
        allow_neighboring_matches: Optional[bool] = False,
        exclusion_factor: Optional[float] = 2.0,
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
        inverse_distance : bool, optional
            Wheter to inverse the computed distance, meaning that the method will return
            the k most dissimilar neighbors instead of the k most similar.
        X_index : np.ndarray, shape=(2,), optional
            If ``X`` is a subsequence of the database given in fit, specify its starting
            index as (i_case, i_timestamp). If specified, this subsequence and the
            neighboring ones (according to ``exclusion_factor``) won't be considered as
            admissible candidates.
        allow_neighboring_matches: bool, optional
            Wheter the top-k candidates can be neighboring subsequences.
        exclusion_factor : float, default=2.
            A factor of the query length used to define the exclusion zone when
            ``allow_neighboring_matches`` is set to False. For a given timestamp,
            the exclusion zone starts from
            :math:`id_timestamp - query_length//exclusion_factor` and end at
            :math:`id_timestamp + query_length//exclusion_factor`.

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
        if self.length != X.shape[1]:
            raise ValueError(
                f"Expected X to be of shape {(self.n_channels_, self.length)} but"
                f" got {X.shape} in find_neighbors."
            )

        X_index = self._check_X_index_array(X_index)
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        neighbors, distances = self._find_neighbors(
            X,
            k=k,
            threshold=threshold,
            inverse_distance=inverse_distance,
            X_index=X_index,
            allow_neighboring_matches=allow_neighboring_matches,
            exclusion_factor=exclusion_factor,
        )
        set_num_threads(prev_threads)
        if len(neighbors) < k:
            warnings.warn(
                f"The number of admissible neighbors found is {len(neighbors)}, instead"
                f" of {k}",
                stacklevel=2,
            )
        return neighbors, distances

    def _check_X_index_int(self, X_index: int):
        """
        Check wheter the X_index parameter is correctly formated and is admissible.

        This check is made for motif search functions.

        Parameters
        ----------
        X_index : int
            Index of a series in X_.

        Returns
        -------
        X_index : int
            Index of a series in X_

        """
        if X_index is not None:
            if not isinstance(X_index, int):
                raise TypeError("Expected an integer for X_index but got {X_index}")

            if X_index >= self.n_cases_ or X_index < 0:
                raise ValueError(
                    "The value of X_index cannot exced the number "
                    "of series in the collection given during fit. Expected a value "
                    f"between [0, {self.n_cases_ - 1}] but got {X_index}"
                )
        return X_index

    def _check_X_index_array(self, X_index: np.ndarray):
        """
        Check wheter the X_index parameter is correctly formated and is admissible.

        This check is made for neighbour search functions.

        Parameters
        ----------
        X_index : np.ndarray, 1D array of shape (2)
            Array of integer containing the sample and timestamp identifiers of the
            starting point of a subsequence in X_.

        Returns
        -------
        X_index : np.ndarray, 1D array of shape (2)
            Array of integer containing the sample and timestamp identifiers of the
            starting point of a subsequence in X_.

        """
        if X_index is not None:
            if (
                isinstance(X_index, list)
                and len(X_index) == 2
                and isinstance(X_index[0], int)
                and isinstance(X_index[1], int)
            ):
                X_index = np.asarray(X_index, dtype=int)
            elif len(X_index) != 2:
                raise TypeError(
                    "Expected a numpy array or list of integers with 2 elements "
                    f"for X_index but got {X_index}"
                )
            elif (
                not (isinstance(X_index[0], int) or not isinstance(X_index[1], int))
                or X_index.dtype != int
            ):
                raise TypeError(
                    "Expected a numpy array or list of integers for X_index but got "
                    f"{X_index}"
                )

            if X_index[0] >= self.n_cases_ or X_index[0] < 0:
                raise ValueError(
                    "The sample ID (first element) of X_index cannot exced the number "
                    "of series in the collection given during fit. Expected a value "
                    f"between [0, {self.n_cases_ - 1}] but got {X_index[0]}"
                )
            _max_timestamp = self.X_[X_index[0]].shape[1] - self.length + 1
            if X_index[1] >= _max_timestamp:
                raise ValueError(
                    "The timestamp ID (second element) of X_index cannot exced the "
                    "number of timestamps minus the length parameter plus one. Expected"
                    f" a value between [0, {_max_timestamp - 1}] but got {X_index[1]}"
                )
        return X_index

    def _compute_mean_std_from_collection(self, X: np.ndarray):
        """
        Compute the mean and std of each subsequence of size ``length`` in X.

        Parameters
        ----------
        X : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
            Collection of series from which we extract mean and stds. If it is an
            unequal length collection, it should be a list of 2d numpy arrays.

        Returns
        -------
        Tuple(np.ndarray, np.ndarray)
            Both array are of shape (n_cases, n_timepoints-length+1, n_channels),
            the first contains the means and the second the stds for each subsequence
            of size ``length`` in X.

        """
        means = []
        stds = []

        for i_x in range(len(X)):
            _mean, _std = sliding_mean_std_one_series(X[i_x], self.length, 1)
            stds.append(_std)
            means.append(_mean)

        if self.metadata_["unequal_length"]:
            return List(means), List(stds)
        else:
            return np.asarray(means), np.asarray(stds)

    def _fit(self, X, y=None):
        if self.length >= self.min_timepoints_ or self.length < 1:
            raise ValueError(
                "The length of the query should be inferior or equal to the length of "
                "data (X_) provided during fit, but got {} for X and {} for X_".format(
                    self.length, self.min_timepoints_
                )
            )

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
        allow_neighboring_matches: Optional[bool] = False,
        exclusion_factor: Optional[float] = 2.0,
    ): ...

    @abstractmethod
    def _find_neighbors(
        self,
        X: np.ndarray,
        k: Optional[int] = 1,
        threshold: Optional[float] = np.inf,
        inverse_distance: Optional[bool] = False,
        X_index=None,
        allow_neighboring_matches: Optional[bool] = False,
        exclusion_factor: Optional[float] = 2.0,
    ): ...

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> dict:
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {"length": 3}


class BaseMatrixProfile(BaseSubsequenceSearch):
    """Base class for Matrix Profile methods using a length parameter."""

    def _find_motifs(
        self,
        X: np.ndarray,
        k: Optional[int] = 1,
        threshold: Optional[float] = np.inf,
        X_index: Optional[int] = None,
        inverse_distance: Optional[bool] = False,
        allow_neighboring_matches: Optional[bool] = False,
        exclusion_factor: Optional[float] = 2.0,
    ):
        exclusion_size = self.length // exclusion_factor

        MP, IP = self.compute_matrix_profile(
            k,
            threshold,
            exclusion_size,
            inverse_distance,
            allow_neighboring_matches,
            X=X,
            X_index=X_index,
        )
        # TODO check motif extraction logic, sure its not this one
        MP_avg = np.array([[np.mean(MP[i]) for i in range(len(MP))]])
        # TODO: appening IP of identified motifs to return to get motifs matches in X_
        return _extract_top_k_from_dist_profile(
            MP_avg,
            k,
            threshold,
            allow_neighboring_matches,
            exclusion_size,
        )

    def _find_neighbors(
        self,
        X: np.ndarray,
        k: Optional[int] = 1,
        threshold: Optional[float] = np.inf,
        inverse_distance: Optional[bool] = False,
        X_index=None,
        allow_neighboring_matches: Optional[bool] = False,
        exclusion_factor: Optional[float] = 2.0,
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
        inverse_distance : bool, optional
            Wheter to inverse the computed distance, meaning that the method will return
            the k most dissimilar neighbors instead of the k most similar.
        X_index : np.ndarray, shape=(2,), optional
            If ``X`` is a subsequence of the database given in fit, specify its starting
            index as (i_case, i_timestamp). If specified, this subsequence and the
            neighboring ones (according to ``exclusion_factor``) won't be considered as
            admissible candidates.
        allow_neighboring_matches: bool, optional
            Wheter the top-k candidates can be neighboring subsequences.
        exclusion_factor : float, default=2.
            A factor of the query length used to define the exclusion zone when
            ``allow_neighboring_matches`` is set to False. For a given timestamp, the
            exclusion zone starts from
            :math:`id_timestamp - query_length//exclusion_factor` and end at
            :math:`id_timestamp + query_length//exclusion_factor`.
        """
        exclusion_size = self.length // exclusion_factor
        dist_profiles = self.compute_distance_profile(X)

        if inverse_distance:
            dist_profiles = _inverse_distance_profile_list(dist_profiles)

        # Deal with self-matches
        if X_index is not None:
            _max_timestamp = self.X_[X_index[0]].shape[1] - self.length
            ub = min(X_index[1] + exclusion_size, _max_timestamp)
            lb = max(0, X_index[1] - exclusion_size)
            dist_profiles[X_index[0]][lb:ub] = np.inf

        return _extract_top_k_from_dist_profile(
            dist_profiles,
            k,
            threshold,
            allow_neighboring_matches,
            exclusion_size,
        )

    @abstractmethod
    def compute_matrix_profile(
        self,
        X: np.ndarray,
        k: int,
        threshold: float,
        exclusion_size: int,
        inverse_distance: bool,
        allow_neighboring_matches: bool,
        X_index: Optional[int] = None,
    ):
        """Compute matrix profiles between X_ and X or between all series in X_."""
        ...

    @abstractmethod
    def compute_distance_profile(self, X: np.ndarray):
        """Compute distrance profiles between X_ and X (a series of size length)."""
        ...
