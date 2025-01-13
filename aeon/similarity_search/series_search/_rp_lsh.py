"""Random projection LSH index."""

import numpy as np
from numba import njit, prange

from aeon.similarity_search.series_search._base import BaseIndexSearch

# TPB = 16


@njit(cache=True)
def _hamming_dist(X, Y):
    d = 0
    for i in prange(X.shape[0]):
        d += X[i] ^ Y[i]
    return d


@njit(cache=True, parallel=True)
def _hamming_dist_matrix(bool_hashes_value_list, bool_hashes):
    n_hash_funcs = bool_hashes.shape[0]
    res = np.zeros((n_hash_funcs, bool_hashes_value_list.shape[0]), dtype=np.int64)
    for i in prange(n_hash_funcs):
        for j in range(bool_hashes_value_list.shape[0]):
            res[i, j] = _hamming_dist(bool_hashes_value_list[j], bool_hashes[i])
    return res


@njit(cache=True, fastmath=True, parallel=True)
def _series_to_bool(X, hash_funcs, start_points, length):
    n_hash_funcs = hash_funcs.shape[0]
    res = np.empty(n_hash_funcs, dtype=np.bool_)
    for j in prange(n_hash_funcs):
        res[j] = _nb_flat_dot(
            X[start_points[j] : start_points[j] + length], hash_funcs[j]
        )
    return res


@njit(cache=True, fastmath=True)
def _nb_flat_dot(X, Y):
    n_channels, n_timepoints = X.shape
    out = 0
    for i in prange(n_channels):
        for j in prange(n_timepoints):
            out += X[i, j] * Y[i, j]
    return out >= 0


@njit(cache=True, parallel=True)
def _collection_to_bool(X, hash_funcs, start_points, length):
    n_hash_funcs = hash_funcs.shape[0]
    n_samples = X.shape[0]
    res = np.empty((n_samples, n_hash_funcs), dtype=np.bool_)
    for j in prange(n_hash_funcs):
        for i in range(n_samples):
            res[i, j] = _nb_flat_dot(
                X[i, :, start_points[j] : start_points[j] + length], hash_funcs[j]
            )

    return res


class RP_LSH_Cosine(BaseIndexSearch):
    """
    Random Projection Locality Sensitive Hashing index for cosine similarity.

    In this method based on SimHash, we define a hash function as a boolean operation
    such as, given a random vector ``V`` of shape ``(n_channels, L)`` and a time series
    ``X`` of shape ``(n_channels, n_timeponts)`` (with ``L<=n_timepoints``), we compute
    ``X.V > 0`` to obtain the boolean result.
    In the case where ``L<n_timepoints``, each hash function is affected a random
    starting point ``s`` (between ``[0, n_timepoints - L]``) to compute the dot product
    as ``X[:, s:s+L].V``

    Note that this method will not provide exact results, but will perform approximate
    searchs. This also ignore any temporal correlation and treat series as
    high dimensional points.

    Parameters
    ----------
    n_hash_funcs : int, optional
        Number of random hashing function to use to index series. The default is 128.
    hash_func_coverage : float, optional
        A value in the interval ]0,1] which defines the size L fo the random vectors
        relative to the size of the input time series. The default is 1.0.
    use_discrete_vectors: bool, optional,
        Wheter to use dicrete vectors with values -1 or 1 as random vector. If false,
        the values of the random vectors are drawn uniformly between [-1,1].
    random_state: int, optional
        A random seed to seed the index building. The default is None.

    Example
    -------
    >>> from aeon.datasets import load_classification
    >>> from aeon.similarity_search.series_search import RP_LSH_Cosine
    >>> index = RP_LSH_Cosine()
    >>> X, y = load_classification("ArrowHead")
    >>> index.fit(X[:200])
    >>> r = index.find_neighbors(X[201])
    """

    _tags = {
        "capability:unequal_length": False,
    }

    def __init__(
        self,
        n_hash_funcs=128,
        hash_func_coverage=0.25,
        use_discrete_vectors=True,
        random_state=None,
        normalise=False,
        n_jobs=1,
    ):
        self.n_hash_funcs = n_hash_funcs
        self.hash_func_coverage = hash_func_coverage
        self.use_discrete_vectors = use_discrete_vectors
        self.random_state = random_state
        super().__init__(normalise=normalise, n_jobs=n_jobs)

    def _build_index(self):
        """
        Build the index based on the data stored in X_.

        Returns
        -------
        self

        """
        rng = np.random.default_rng(self.random_state)
        n_timepoints = self.X_.shape[2]
        self.window_length_ = max(1, int(n_timepoints * self.hash_func_coverage))

        if self.use_discrete_vectors:
            self.hash_funcs_ = rng.choice(
                [-1, 1], size=(self.n_hash_funcs, self.n_channels_, self.window_length_)
            )
        else:
            self.hash_funcs_ = rng.uniform(
                low=-1,
                high=1.0,
                size=(self.n_hash_funcs, self.n_channels_, self.window_length_),
            )
        self.start_points_ = rng.choice(
            n_timepoints - self.window_length_ + 1,
            size=self.n_hash_funcs,
            replace=True,
        )

        bool_hashes = _collection_to_bool(
            self.X_, self.hash_funcs_, self.start_points_, self.window_length_
        )

        str_hashes = [hash(bool_hashes[i].tobytes()) for i in range(len(bool_hashes))]
        self.dict_X_index_ = {}
        self.dict_bool_hashes_ = {}
        for i in range(len(str_hashes)):
            if str_hashes[i] in self.dict_X_index_:
                self.dict_X_index_[str_hashes[i]].append(i)
            else:
                self.dict_X_index_[str_hashes[i]] = [i]
                self.dict_bool_hashes_[str_hashes[i]] = bool_hashes[i]

        self.bool_hashes_value_list_ = np.asarray(list(self.dict_bool_hashes_.values()))
        self.bool_hashes_key_list_ = np.asarray(list(self.dict_bool_hashes_.keys()))

        return self

    def _get_bucket_content(self, key):
        return self.dict_X_index_[key]

    def _get_bucket_sizes(self):
        return {key: len(self.dict_X_index_[key]) for key in self.dict_X_index_}

    def _get_series_bucket(self, X):
        """
        Get the matching bucket of a single series X if it exists in the index.

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        bool_hash = _series_to_bool(
            X, self.hash_funcs_, self.start_points_, self.window_length_
        )
        str_hash = hash(bool_hash.tobytes())
        if str_hash in self.dict_X_index_:
            return str_hash
        else:
            return None

    def _query_index(
        self,
        X,
        k=1,
        threshold=np.inf,
        inverse_distance=False,
    ):
        """
        Find approximate nearest neighbors of a collection in the index.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_tiempoints)
            Series for which we want to find neighbors.
        k : int, optional
            Number of neighbors to return for each series. The default is 1.
        threshold : int, optional
            A threshold on the distance to determine which candidates will be returned.
        inverse_distance : bool, optional
            Wheter to inverse the computed distance, meaning that the method will return
            the k most dissimilar neighbors instead of the k most similar.

        Returns
        -------
        top_k : np.ndarray, shape = (n_cases, k)
            Indexes of k series in X_ (the index) that are close to each series in X.
        top_k_dist : np.ndarray, shape = (n_cases, k)
            Distance of k series in X_ (the index) to each series in X. The distance
            is the hamming distance between the result of each hash function.
        """
        bool_hashes = _series_to_bool(
            X, self.hash_funcs_, self.start_points_, self.window_length_
        )
        top_k = np.zeros(k, dtype=int)
        top_k_dist = np.zeros(k, dtype=float)
        dists = _hamming_dist_matrix(
            self.bool_hashes_value_list_, bool_hashes[np.newaxis, :]
        )[0]
        if inverse_distance:
            dists = 1 / (dists + 1e-8)
        # Get top k buckets
        ids = np.argpartition(dists, kth=k)[:k]
        # and reoder them
        ids = ids[np.argsort(dists[ids])]

        _i_bucket = 0
        current_k = 0
        while current_k < k:
            if dists[ids[_i_bucket]] <= threshold:
                candidates = self.dict_X_index_[
                    self.bool_hashes_key_list_[ids[_i_bucket]]
                ]
                # Can do exact search by computing distances here
                if len(candidates) > k - current_k:
                    candidates = candidates[: k - current_k]
                top_k[current_k : current_k + len(candidates)] = candidates
                top_k_dist[current_k : current_k + len(candidates)] = dists[
                    ids[_i_bucket]
                ]
                current_k += len(candidates)
            else:
                break
            _i_bucket += 1
        return top_k[:current_k], top_k_dist[:current_k]
