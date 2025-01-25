"""Implementation of STOMP with squared euclidean distance."""

__maintainer__ = ["baraline"]
__all__ = ["StompMotif"]

from typing import Optional

import numpy as np
from numba import njit
from numba.typed import List

from aeon.similarity_search.series._base import BaseSeriesSimilaritySearch
from aeon.similarity_search.series._commons import (
    _extract_top_k_from_dist_profile,
    _extract_top_k_motifs,
    _extract_top_r_motifs,
    _inverse_distance_profile,
    _update_dot_products,
    get_ith_products,
)
from aeon.similarity_search.series.neighbors._mass import (
    _normalized_squared_distance_profile,
    _squared_distance_profile,
)
from aeon.utils.numba.general import sliding_mean_std_one_series


class StompMotif(BaseSeriesSimilaritySearch):
    """
    Estimator to extract top k motifs using STOMP, descibed in [1]_.

    This estimators allows to perform multiple type of motif search operations by using
    different parameterization. We base oursleves on Figure 3 of [2]_ to establish the
    following list, we do not yet support "Learning" and "Valmod" motifs :

        - for "Pair Motifs" : This is the default configuration

        - for "k-Motiflets" : {
            "motif_size": k,
        }

        - for "k-motifs" (naming is confusing here, it is a range based motif): {
            "motif_size":np.inf,
            "dist_threshold":r,
            "motif_extraction_method":"r_motifs"
        }

    Parameters
    ----------
    length : int
        The length of the motifs to extract. This is the length of the subsequence
        that will be used in the computations.
    normalize : bool
        Wheter the computations between subsequences should use a z-normalied distance.

    Notes
    -----
    This estimator only provide exact computation method, faster approximate methods
    also exists in the litterature. We use a squared euclidean distance instead of the
    euclidean distance, if you want euclidean distance results, you should square root
    the obtained results.

    References
    ----------
    .. [1] Yan Zhu, Zachary Zimmerman, Nader Shakibay Senobari, Chin-Chia Michael
    Yeh, Gareth Funning, Abdullah Mueen, Philip Brisk, and Eamonn Keogh. 2016.
    Matrix profile II: Exploiting a novel algorithm and GPUs to break the one hundred
    million barrier for time series motifs and joins. In 2016 IEEE 16th international
    conference on data mining (ICDM). IEEE, 739–748.
    .. [2] Patrick Schäfer and Ulf Leser. 2022. Motiflets: Simple and Accurate Detection
    of Motifs in Time Series. Proc. VLDB Endow. 16, 4 (December 2022), 725–737.
    https://doi.org/10.14778/3574245.3574257
    """

    def __init__(
        self,
        length: int,
        normalize: Optional[bool] = False,
    ):
        self.length = length
        self.normalize = normalize
        super().__init__()

    def _fit(
        self,
        X: np.ndarray,
        y=None,
    ):
        if self.normalize:
            self.X_means_, self.X_stds_ = sliding_mean_std_one_series(X, self.length, 1)
        return self

    def predict(
        self,
        X: np.ndarray = None,
        k: Optional[int] = 1,
        motif_size: Optional[int] = 1,
        dist_threshold: Optional[float] = np.inf,
        allow_trivial_matches: Optional[bool] = False,
        exclusion_factor: Optional[float] = 2,
        inverse_distance: Optional[bool] = False,
        motif_extraction_method: Optional[str] = "k_motifs",
    ):
        """
        Exctract the motifs of X_ relative to a series X using STOMP matrix prfoile.

        To compute self-motifs, X is set to None.

        Parameters
        ----------
        X : np.ndarray, shape=(n_channels, n_timepoint)
            Series to use to compute the matrix profile against X_. If None, will
            compute the self matrix profile of X_. Motifs will then be extracted from
            the matrix profile.
        k : int
            The number of motifs to return. The default is 1, meaning we return only
            the motif set with the minimal sum of distances to its query.
        motif_size : int
            The number of subsequences in a motif. Default is 1, meaning we extract
            motif pairs (the query and its best match)
        dist_threshold : float
            The maximum allowed distance of a candidate subsequence of X to a query
            subsequence from X_ for the candidate to be considered as a neighbor.
        allow_trivial_matches: bool, optional
            Wheter a neighbors of a match to a query can be also considered as matches
            (True), or if an exclusion zone is applied around each match to avoid
            trivial matches with their direct neighbors (False).
        exclusion_factor : float, default=1.
            A factor of the query length used to define the exclusion zone when
            ``allow_trivial_matches`` is set to False. For a given timestamp,
            the exclusion zone starts from
            :math:`id_timestamp - length//exclusion_factor` and end at
            :math:`id_timestamp + length//exclusion_factor`.
        inverse_distance : bool
            If True, the matching will be made on the inverse of the distance, and thus,
            the farther neighbors will be returned instead of the closest ones.
        motif_extraction_method : str
            A string indicating the methodology to use to extract the top motifs.
            Available methods are "r_motifs" and "k_motifs". "r_motifs" means we rank
            motif set by their cardinality, with higher is better. "k_motifs" means
            we rank motif set by their maximum distance to their query

        Returns
        -------
        np.ndarray, shape = (k, motif_size)
            The indexes of the best matches in ``distance_profile``.
        np.ndarray, shape = (k, motif_size)
            The distances of the best matches.

        """
        X = self._pre_predict(X)
        if motif_extraction_method not in ["k_motifs", "r_motifs"]:
            raise ValueError(
                "Expected motif_extraction_method to be either 'k_motifs' or 'r_motifs'"
                f"but got {motif_extraction_method}"
            )

        MP, IP = self.compute_matrix_profile(
            X,
            motif_size=motif_size,
            dist_threshold=dist_threshold,
            allow_trivial_matches=allow_trivial_matches,
            exclusion_factor=exclusion_factor,
            inverse_distance=inverse_distance,
        )
        if motif_extraction_method == "k_motifs":
            return _extract_top_k_motifs(
                MP, IP, k, allow_trivial_matches, self.length // exclusion_factor
            )
        elif motif_extraction_method == "r_motifs":
            return _extract_top_r_motifs(
                MP, IP, k, allow_trivial_matches, self.length // exclusion_factor
            )

    def compute_matrix_profile(
        self,
        X: np.ndarray = None,
        motif_size: Optional[int] = 1,
        dist_threshold: Optional[float] = np.inf,
        allow_trivial_matches: Optional[bool] = False,
        exclusion_factor: Optional[float] = 2,
        inverse_distance: Optional[bool] = False,
    ):
        """
        Compute matrix profile.

        The matrix profile is computed on the series given in fit (X_). If X is
        not given, computes the self matrix profile of X_. Otherwise, compute the matrix
        profile of X_ relative to X.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_timepoints)
            A 2D array time series on against which the matrix profile of X_ will be
            computed.
        motif_size : int
            The number of subsequences in a motif. Default is 1, meaning we extract
            motif pairs (the query and its best match).
        dist_threshold : float
           The maximum allowed distance of a candidate subsequence of X to a query
           subsequence from X_ for the candidate to be considered as a neighbor.
        inverse_distance : bool
            If True, the matching will be made on the inverse of the distance, and thus,
            the worst matches to the query will be returned instead of the best ones.
        exclusion_factor : float, default=1.
            A factor of the query length used to define the exclusion zone when
            ``allow_trivial_matches`` is set to False. For a given timestamp,
            the exclusion zone starts from
            :math:`id_timestamp - length//exclusion_factor` and end at
            :math:`id_timestamp + length//exclusion_factor`.

        Returns
        -------
        MP : TypedList of np.ndarray (n_timepoints - L + 1)
            Matrix profile distances for each query subsequence. n_timepoints is the
            number of timepoint of X_. Each element of the list contains array of
            variable size.
        IP : TypedList of np.ndarray (n_timepoints - L + 1)
            Indexes of the top matches for each query subsequence. n_timepoints is the
            number of timepoint of X_. Each element of the list contains array of
            variable size.
        """
        if X is None:
            is_self_mp = True
            X = self.X_
            if self.normalize:
                X_means, X_stds = self.X_means_, self.X_stds_
        else:
            is_self_mp = False
            if self.normalize:
                X_means, X_stds = sliding_mean_std_one_series(X, self.length, 1)
        X_dotX = get_ith_products(X, self.X_, self.length, 0)
        exclusion_size = self.length // exclusion_factor

        if motif_size == np.inf:
            # convert infs here as numba seem to not be able to do == np.inf ?
            motif_size = X.shape[1] - self.length + 1

        if self.normalize:
            MP, IP = _stomp_normalized(
                self.X_,
                X,
                X_dotX,
                self.X_means_,
                self.X_stds_,
                X_means,
                X_stds,
                self.length,
                motif_size,
                dist_threshold,
                allow_trivial_matches,
                exclusion_size,
                inverse_distance,
                is_self_mp,
            )
        else:
            MP, IP = _stomp(
                self.X_,
                X,
                X_dotX,
                self.length,
                motif_size,
                dist_threshold,
                allow_trivial_matches,
                exclusion_size,
                inverse_distance,
                is_self_mp,
            )
        return MP, IP

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        if parameter_set == "default":
            params = {"length": 3}
        else:
            raise NotImplementedError(
                f"The parameter set {parameter_set} is not yet implemented"
            )
        return params


@njit(cache=True, fastmath=True)
def _stomp_normalized(
    X_A,
    X_B,
    AdotB,
    X_A_means,
    X_A_stds,
    X_B_means,
    X_B_stds,
    L,
    motif_size,
    dist_threshold,
    allow_trivial_matches,
    exclusion_size,
    inverse_distance,
    is_self_mp,
):
    """
    Compute the Matrix Profile using the STOMP algorithm with normalized distances.

    X_A : np.ndarray, 2D array of shape (n_channels, n_timepoints)
        The series from which the queries will be extracted.
    X_B : np.ndarray, 2D array of shape (n_channels, series_length)
        The time series on which the distance profile of each query will be computed.
    AdotB : np.ndarray, 2D array of shape (n_channels, series_length - L + 1)
        Precomputed dot products between the first query of size L of X_A and X_B.
    X_A_means : np.ndarray, 2D array of shape (n_channels, n_timepoints - L + 1)
        Means of each subsequences of X_A of size L.
    X_A_stds : np.ndarray, 2D array of shape (n_channels, n_timepoints - L + 1)
        Stds of each subsequences of X of size L.
    X_B_means : np.ndarray, 2D array of shape (n_channels, series_length - L + 1)
        Means of each subsequences of X_B of size L.
    X_B_stds : np.ndarray, 2D array of shape (n_channels, series_length - L + 1)
        Stds of each subsequences of X_B of size L.
    L : int
        Length of the subsequences used for the distance computation.
    motif_size : int
        The number of subsequences to extract from each distance profile.
    dist_threshold : float
       The maximum allowed distance of a candidate subsequence of X to a query
       subsequence from X_ for the candidate to be considered as a neighbor.
    allow_trivial_matches : bool
        Wheter the top-k candidates can be neighboring subsequences.
    exclusion_size : int
        The size of the exclusion zone used to prevent returning as top k candidates
        the ones that are close to each other (for example i and i+1).
        It is used to define a region between
        :math:`id_timestamp - exclusion_size` and
        :math:`id_timestamp + exclusion_size` which cannot be returned
        as best match if :math:`id_timestamp` was already selected. By default,
        the value None means that this is not used.
    inverse_distance : bool
        If True, the matching will be made on the inverse of the distance, and thus, the
        worst matches to the query will be returned instead of the best ones.
    is_self_mp : bool
        Wheter X_A == X_B.

    Returns
    -------
    MP : TypedList of np.ndarray (n_timepoints - L + 1)
        Matrix profile distances for each query subsequence. n_timepoints is the
        number of timepoint of X_. Each element of the list contains array of
        variable size.
    IP : TypedList of np.ndarray (n_timepoints - L + 1)
        Indexes of the top matches for each query subsequence. n_timepoints is the
        number of timepoint of X_. Each element of the list contains array of
        variable size.
    """
    n_queries = X_A.shape[1] - L + 1
    _max_timestamp = X_B.shape[1] - L + 1
    MP = List()
    IP = List()

    for i_q in range(n_queries):
        # size T.shape[1] - L + 1
        dist_profile = _normalized_squared_distance_profile(
            AdotB, X_B_means, X_B_stds, X_A_means[:, i_q], X_A_stds[:, i_q], L
        )

        if i_q + 1 < n_queries:
            AdotB = _update_dot_products(X_B, X_A, AdotB, L, i_q + 1)

        if inverse_distance:
            dist_profile = _inverse_distance_profile(dist_profile)

        if is_self_mp:
            ub = min(i_q + exclusion_size, _max_timestamp + 1)
            lb = max(0, i_q - exclusion_size)
            dist_profile[lb:ub] = np.inf

        _top_indexes, top_dists = _extract_top_k_from_dist_profile(
            dist_profile,
            motif_size,
            dist_threshold,
            allow_trivial_matches,
            exclusion_size,
        )
        top_indexes = np.zeros((len(_top_indexes), 2), dtype=np.int64)
        for i_idx in range(len(_top_indexes)):
            top_indexes[i_idx, 0] = i_q
            top_indexes[i_idx, 1] = _top_indexes[i_idx]
        MP.append(top_dists)
        IP.append(top_indexes)

    return MP, IP


@njit(cache=True, fastmath=True)
def _stomp(
    X_A,
    X_B,
    AdotB,
    L,
    motif_size,
    dist_threshold,
    allow_trivial_matches,
    exclusion_size,
    inverse_distance,
    is_self_mp,
):
    """
    Compute the Matrix Profile using the STOMP algorithm with non-normalized distances.

    X_A : np.ndarray, 2D array of shape (n_channels, n_timepoints)
        The series from which the queries will be extracted.
    X_B : np.ndarray, 2D array of shape (n_channels, series_length)
        The time series on which the distance profile of each query will be computed.
    AdotB : np.ndarray, 2D array of shape (n_channels, series_length - L + 1)
        Precomputed dot products between the first query of size L of X_A and X_B.
    L : int
        Length of the subsequences used for the distance computation.
    motif_size : int
        The number of subsequences to extract from each distance profile.
    dist_threshold : float
       The maximum allowed distance of a candidate subsequence of X to a query
       subsequence from X_ for the candidate to be considered as a neighbor.
    allow_trivial_matches : bool
        Wheter the top-k candidates can be neighboring subsequences.
    exclusion_size : int
        The size of the exclusion zone used to prevent returning as top k candidates
        the ones that are close to each other (for example i and i+1).
        It is used to define a region between
        :math:`id_timestamp - exclusion_size` and
        :math:`id_timestamp + exclusion_size` which cannot be returned
        as best match if :math:`id_timestamp` was already selected. By default,
        the value None means that this is not used.
    inverse_distance : bool
        If True, the matching will be made on the inverse of the distance, and thus, the
        worst matches to the query will be returned instead of the best ones.
    is_self_mp : bool
        Wheter X_A == X_B.

    Returns
    -------
    MP : TypedList of np.ndarray (n_timepoints - L + 1)
        Matrix profile distances for each query subsequence. n_timepoints is the
        number of timepoint of X_. Each element of the list contains array of
        variable size.
    IP : TypedList of np.ndarray (n_timepoints - L + 1)
        Indexes of the top matches for each query subsequence. n_timepoints is the
        number of timepoint of X_. Each element of the list contains array of
        variable size.
    """
    n_queries = X_A.shape[1] - L + 1
    _max_timestamp = X_B.shape[1] - L + 1
    MP = List()
    IP = List()

    # For each query of size L in X_A
    for i_q in range(n_queries):
        Q = X_A[:, i_q : i_q + L]
        dist_profile = _squared_distance_profile(AdotB, X_B, Q)
        if i_q + 1 < n_queries:
            AdotB = _update_dot_products(X_B, X_A, AdotB, L, i_q + 1)

        if inverse_distance:
            dist_profile = _inverse_distance_profile(dist_profile)

        if is_self_mp:
            ub = min(i_q + exclusion_size, _max_timestamp + 1)
            lb = max(0, i_q - exclusion_size)
            dist_profile[lb:ub] = np.inf

        _top_indexes, top_dists = _extract_top_k_from_dist_profile(
            dist_profile,
            motif_size,
            dist_threshold,
            allow_trivial_matches,
            exclusion_size,
        )
        top_indexes = np.zeros((len(_top_indexes), 2), dtype=np.int64)
        for i_idx in range(len(_top_indexes)):
            top_indexes[i_idx, 0] = i_q
            top_indexes[i_idx, 1] = _top_indexes[i_idx]
        MP.append(top_dists)
        IP.append(top_indexes)

    return MP, IP
