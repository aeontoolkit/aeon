"""Base class for query search."""

__maintainer__ = ["baraline"]

from typing import Optional, final

import numpy as np
from numba import get_num_threads, set_num_threads

from aeon.similarity_search._commons import (
    extract_top_k_and_threshold_from_distance_profiles,
)
from aeon.similarity_search.base import BaseSimilaritySearch
from aeon.similarity_search.distance_profiles.euclidean_distance_profile import (
    euclidean_distance_profile,
    normalised_euclidean_distance_profile,
)
from aeon.similarity_search.distance_profiles.squared_distance_profile import (
    normalised_squared_distance_profile,
    squared_distance_profile,
)


class QuerySearch(BaseSimilaritySearch):
    """
    Query search estimator.

    The query search estimator will return a set of matches of a query in a search space
    , which is defined by a time series dataset given during fit. Depending on the `k`
    and/or `threshold` parameters, which condition what is considered a valid match
    during the search, the number of matches will vary. If `k` is used, at most `k`
    matches (the `k` best) will be returned, if `threshold` is used and `k` is set to
    `np.inf`, all the candidates which distance to the query is inferior or equal to
    `threshold` will be returned. If both are used, the `k` best matches to the query
    with distance inferior to `threshold` will be returned.


    Parameters
    ----------
    k : int, default=1
        The number of best matches to return during predict for a given query.
    threshold : float, default=np.inf
        The number of best matches to return during predict for a given query.
    distance : str, default="euclidean"
        Name of the distance function to use. A list of valid strings can be found in
        the documentation for :func:`aeon.distances.get_distance_function`.
        If a callable is passed it must either be a python function or numba function
        with nopython=True, that takes two 1d numpy arrays as input and returns a float.
    distance_args : dict, default=None
        Optional keyword arguments for the distance function.
    normalise : bool, default=False
        Whether the distance function should be z-normalised.
    speed_up : str, default='fastest'
        Which speed up technique to use with for the selected distance
        function. By default, the fastest algorithm is used. A list of available
        algorithm for each distance can be obtained by calling the
        `get_speedup_function_names` function.
    inverse_distance : bool, default=False
        If True, the matching will be made on the inverse of the distance, and thus, the
        worst matches to the query will be returned instead of the best ones.
    n_jobs : int, default=1
        Number of parallel jobs to use.
    store_distance_profiles : bool, default=False.
        Whether to store the computed distance profiles in the attribute
        "distance_profiles_" after calling the predict method. It will store the raw
        distance profile, meaning without potential inversion or thresholding applied.

    Attributes
    ----------
    X_ : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input time series stored during the fit method. This is the
        database we search in when given a query.
    distance_profile_function : function
        The function used to compute the distance profile. This is determined
        during the fit method based on the distance and normalise
        parameters.

    Notes
    -----
    For now, the multivariate case is only treated as independent.
    Distances are computed for each channel independently and then
    summed together.
    """

    def __init__(
        self,
        k: int = 1,
        threshold: float = np.inf,
        distance: str = "euclidean",
        distance_args: Optional[dict] = None,
        inverse_distance: bool = False,
        normalise: bool = False,
        speed_up: str = "fastest",
        n_jobs: int = 1,
        store_distance_profiles: bool = False,
    ):
        self.k = k
        self.threshold = threshold
        self.store_distance_profiles = store_distance_profiles
        self._previous_query_length = -1
        self.axis = 1

        super().__init__(
            distance=distance,
            distance_args=distance_args,
            inverse_distance=inverse_distance,
            normalise=normalise,
            speed_up=speed_up,
            n_jobs=n_jobs,
        )

    def _fit(self, X: np.ndarray, y=None):
        """
        Check input format and store it to be used as search space during predict.

        Parameters
        ----------
        X : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
            Input array to used as database for the similarity search
        y : optional
            Not used.

        Raises
        ------
        TypeError
            If the input X array is not 3D raise an error.

        Returns
        -------
        self

        """
        self.X_ = X
        self.distance_profile_function_ = self._get_distance_profile_function()
        return self

    @final
    def predict(
        self,
        X: np.ndarray,
        axis=1,
        X_index=None,
        exclusion_factor=2.0,
        apply_exclusion_to_result=False,
    ) -> np.ndarray:
        """
        Predict method : Check the shape of X and call _predict to perform the search.

        If the distance profile function is normalised, it stores the mean and stds
        from X and X_, with X_ the training data.

        Parameters
        ----------
        X : np.ndarray, 2D array of shape (n_channels, query_length)
            Input query used for similarity search.
        axis : int
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point. i.e. the
            shape of the data is ``(n_timepoints,n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels,n_timepoints)``.
        X_index : Iterable
            An Interable (tuple, list, array) of length two used to specify the index of
            the query X if it was extracted from the input data X given during the fit
            method. Given the tuple (id_sample, id_timestamp), the similarity search
            will define an exclusion zone around the X_index in order to avoid matching
            X with itself. If None, it is considered that the query is not extracted
            from X_.
        exclusion_factor : float, default=2.
            The factor to apply to the query length to define the exclusion zone. The
            exclusion zone is define from
            :math:`id_timestamp - query_length//exclusion_factor` to
            :math:`id_timestamp + query_length//exclusion_factor`. This also applies to
            the matching conditions defined by child classes. For example, with
            TopKSimilaritySearch, the k best matches are also subject to the exclusion
            zone, but with :math:`id_timestamp` the index of one of the k matches.
        apply_exclusion_to_result : bool, default=False
            Wheter to apply the exclusion factor to the output of the similarity search.
            This means that two matches of the query from the same sample must be at
            least spaced by +/- :math:`query_length//exclusion_factor`.
            This can avoid pathological matching where, for example if we extract the
            best two matches, there is a high chance that if the best match is located
            at :math:`id_timestamp`, the second best match will be located at
            :math:`id_timestamp` +/- 1, as they both share all their values except one.

        Raises
        ------
        TypeError
            If the input X array is not 2D raise an error.
        ValueError
            If the length of the query is greater

        Returns
        -------
        Tuple(ndarray, ndarray)
            The first array, of shape ``(n_matches)``, contains the distance between
            the query and its best matches in X_. The second array, of shape
            ``(n_matches, 2)``, contains the indexes of these matches as
            ``(id_sample, id_timepoint)``. The corresponding match can be
            retrieved as ``X_[id_sample, :, id_timepoint : id_timepoint + length]``.

        """
        self._check_is_fitted()
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)

        query_dim, query_length = self._check_query_format(X, axis)

        mask = self._init_X_index_mask(
            X_index,
            query_length,
            exclusion_factor=exclusion_factor,
        )

        if self.normalise:
            self.query_means_ = np.mean(X, axis=-1)
            self.query_stds_ = np.std(X, axis=-1)
            if self._previous_query_length != query_length:
                self._store_mean_std_from_inputs(query_length)

        if apply_exclusion_to_result:
            exclusion_size = query_length // exclusion_factor
        else:
            exclusion_size = None

        self._previous_query_length = query_length

        X_preds = self._predict(
            self._call_distance_profile(X, mask),
            exclusion_size=exclusion_size,
        )
        set_num_threads(prev_threads)
        return X_preds

    def _predict(
        self, distance_profiles: np.ndarray, exclusion_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Private predict method for QuerySearch.

        It takes the distance profiles and apply the `k` and `threshold` conditions to
        return the set of best matches.

        Parameters
        ----------
        distance_profiles : np.ndarray, 2D array of shape (n_cases, n_timepoints - query_length + 1)  # noqa: E501
            Precomputed distance profile.
        exclusion_size : int, optional
            The size of the exclusion zone used to prevent returning as top k candidates
            the ones that are close to each other (for example i and i+1).
            It is used to define a region between
            :math:`id_timestamp - exclusion_size` and
            :math:`id_timestamp + exclusion_size` which cannot be returned
            as best match if :math:`id_timestamp` was already selected. By default,
            the value None means that this is not used.

        Returns
        -------
        Tuple(ndarray, ndarray)
            The first array, of shape ``(n_matches)``, contains the distance between
            the query and its best matches in X_. The second array, of shape
            ``(n_matches, 2)``, contains the indexes of these matches as
            ``(id_sample, id_timepoint)``. The corresponding match can be
            retrieved as ``X_[id_sample, :, id_timepoint : id_timepoint + length]``.


        """
        if self.store_distance_profiles:
            self.distance_profiles_ = distance_profiles
        # Define id sample and timestamp to not "loose" them due to concatenation
        return extract_top_k_and_threshold_from_distance_profiles(
            distance_profiles,
            k=self.k,
            threshold=self.threshold,
            exclusion_size=exclusion_size,
            inverse_distance=self.inverse_distance,
        )

    def _check_query_format(self, X, axis):
        if axis not in [0, 1]:
            raise ValueError("The axis argument is expected to be either 1 or 0")
        if self.axis != axis:
            X = X.T
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise TypeError(
                "Error, only supports 2D numpy for now. If the query X is univariate "
                "do X = X[np.newaxis, :]."
            )

        query_dim, query_length = X.shape
        if query_length >= self.min_timepoints_:
            raise ValueError(
                "The length of the query should be inferior or equal to the length of "
                "data (X_) provided during fit, but got {} for X and {} for X_".format(
                    query_length, self.min_timepoints_
                )
            )

        if query_dim != self.n_channels_:
            raise ValueError(
                "The number of feature should be the same for the query X and the data "
                "(X_) provided during fit, but got {} for X and {} for X_".format(
                    query_dim, self.n_channels_
                )
            )
        return query_dim, query_length

    def _get_distance_profile_function(self):
        """
        Given distance and speed_up parameters, return the distance profile function.

        Raises
        ------
        ValueError
            If the distance parameter given at initialization is not a string nor a
            numba function or a callable, or if the speedup parameter is unknow or
            unsupported, raisea ValueError.

        Returns
        -------
        function
            The distance profile function matching the distance argument.

        """
        if isinstance(self.distance, str):
            distance_dict = _QUERY_SEARCH_SPEED_UP_DICT.get(self.distance)
            if distance_dict is None:
                raise NotImplementedError(
                    f"No distance profile have been implemented for {self.distance}."
                )
            else:
                speed_up_profile = distance_dict.get(self.normalise).get(self.speed_up)

                if speed_up_profile is None:
                    raise ValueError(
                        f"Unknown or unsupported speed up {self.speed_up} for "
                        f"{self.distance} distance function with"
                    )
                self.speed_up_ = self.speed_up
                return speed_up_profile
        else:
            raise ValueError(
                f"Expected distance argument to be str but got {type(self.distance)}"
            )

    def _call_distance_profile(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Obtain the distance profile function and call it with the query and the mask.

        Parameters
        ----------
        X : np.ndarray, 2D array of shape (n_channels, query_length)
            Input query used for similarity search.
        mask : np.ndarray, 2D array of shape (n_cases, n_timepoints - query_length + 1)
            Boolean array which indicates the candidates that should be evaluated in
            the similarity search.

        Returns
        -------
        distance_profiles : np.ndarray, 2D array of shape (n_cases, n_timepoints - query_length + 1)  # noqa: E501
            The distance profiles between the input time series and the query.

        """
        if self.normalise:
            distance_profiles = self.distance_profile_function_(
                self.X_,
                X,
                mask,
                self.X_means_,
                self.X_stds_,
                self.query_means_,
                self.query_stds_,
            )
        else:
            distance_profiles = self.distance_profile_function_(self.X_, X, mask)

        return distance_profiles

    @classmethod
    def get_speedup_function_names(self) -> dict:
        """
        Get available speedup for query search in aeon.

        The returned structure is a dictionnary that contains the names of all
        avaialble speedups for normalised and non-normalised distance functions.

        Returns
        -------
        dict
            The available speedups name that can be used as parameters in
            similarity search classes.

        """
        speedups = {}
        for dist_name in _QUERY_SEARCH_SPEED_UP_DICT.keys():
            for normalise in _QUERY_SEARCH_SPEED_UP_DICT[dist_name].keys():
                speedups_names = list(
                    _QUERY_SEARCH_SPEED_UP_DICT[dist_name][normalise].keys()
                )
                if normalise:
                    speedups.update({f"normalised {dist_name}": speedups_names})
                else:
                    speedups.update({f"{dist_name}": speedups_names})
        return speedups


_QUERY_SEARCH_SPEED_UP_DICT = {
    "euclidean": {
        True: {
            "fastest": normalised_euclidean_distance_profile,
            "Mueen": normalised_euclidean_distance_profile,
        },
        False: {
            "fastest": euclidean_distance_profile,
            "Mueen": euclidean_distance_profile,
        },
    },
    "squared": {
        True: {
            "fastest": normalised_squared_distance_profile,
            "Mueen": normalised_squared_distance_profile,
        },
        False: {
            "fastest": squared_distance_profile,
            "Mueen": squared_distance_profile,
        },
    },
}
