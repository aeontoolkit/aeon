from typing import Callable, Optional, Union

import numpy as np
from numpy.random import RandomState
from sklearn.utils.random import check_random_state

from aeon.clustering.base import BaseClusterer
from aeon.distances import get_alignment_path_function, pairwise_distance
from aeon.utils.tags.enum_tags import AlgorithmType

VALID_ELASTIC_SOM_METRICS = [
    "dtw",
    "adtw",
    "edr",
    "erp",
    "msm",
    "shape_dtw",
    "wdtw",
    "twe",
]


class ElasticSOM(BaseClusterer):
    """Elastic Self-Organising Map (SOM) clustering algorithm.

    Self-Organising Maps (SOM) [1]_ are a type of neural network which represents each
    time point in the input time series as a neuron. Each neuron is connected to
    n_cluster output neurons, where n_clusters is the number of clusters. Each
    output neuron represents a cluster. Every input neuron is connected to each output
    neuron using a weight vector. The weights of the connections between input neurons
    and output neurons are learned during training. The algorithm is iterative,
    where each iteration updates the weights of the connections between input neurons
    and output neurons.

    SOM has been adapted to use elastic distances [2]_. The adaptation is done by
    using an elastic distance function to compute the distance between the weights and
    a given time series to find the best matching unit (BMU). Secondly when updating
    the weights an alignment path is used to compute the best alignment between the
    weights and the time series that will update the weights. This means that the
    weights are updated accounting for the alignment path making the updating of the
    weights more efficient.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    distance : str or Callable, default='dtw'
        Distance method to compute similarity between time series. A list of valid
        strings for measures can be found in the documentation for
        :func:`aeon.distances.get_distance_function`. If a callable is passed it must be
        a function that takes two 2d numpy arrays as input and returns a float.
    init : str or np.ndarray, default='random'
        Random is the default and simply chooses k time series at random as
        centroids. It is fast but sometimes yields sub-optimal clustering.
        Kmeans++ [2] and is slower but often more
        accurate than random. It works by choosing centroids that are distant
        from one another.
        First is the fastest method and simply chooses the first k time series as
        centroids.
        If a np.ndarray provided it must be of shape (n_clusters, n_channels,
        n_timepoints)
        and contains the time series to use as centroids.
    sigma : float, default=1.0
        Spread of the neighborhood function.
    learning_rate : float, default=0.5
        Initial learning rate. For a given iterations the learning rate is:
        learning_rate = learning_rate / (1 + iterations / max_iter)
    decay_function : Union[Callable, str], default='asymptotic_decay'
        Decay function to use for the learning rate. Valid strings are:
        ['asymptotic_decay', 'inverse_decay', 'linear_decay'].
        If a Callable is provided must take the form
        Callable[[float, int, int], float], where the first argument is the
        learning rate, the second argument is the current iteration, and the third
        argument is the maximum number of iterations.
    neighborhood_function : Union[Callable, str], default='gaussian'
        Neighborhood function that weights the neighborhood of each time series.
        Valid strings are: ['gaussian', 'mexican_hat'].
        If a Callable is provided must take the form
        Callable[[np.ndarray, np.ndarray, float], np.ndarray], where the first
        argument is the output neuron positions (i.e. which cluster each output
        neuron maps to), the second argument is the index of the best matching
        unit (i.e. which is the closest output neuron), and the third argument is the
        sigma value.
    sigma_decay_function : Union[Callable, str], default='asymptotic_decay'
        Function that reduces sigma each iteration. Valid strings are:
        ['asymptotic_decay', 'inverse_decay', 'linear_decay'].
        If a Callable is provided must take the form
        Callable[[float, int, int], float], where the first argument is the current
        sigma value, the second argument is the current iteration, and the third
        argument is the maximum number of iterations.
    num_iterations : int, default=500
        Number of iterations to run the algorithm for each time series. The recommended
        value is 500 times the number of neurons in the network. Therefore the number
        of iterations is 500 * n_timepoints (as input neurons = n_timepoints).
    distance_params : dict, default=None
        Dictionary containing kwargs for the distance being used. For example if you
        wanted to specify a window for DTW you would pass
        distance_params={"window": 0.2}. See documentation of aeon.distances for more
        details.
    random_state : int, np.random.RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
        If `int`, random_state is the seed used by the random number generator;
        If `np.random.RandomState` instance,
        random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    custom_alignment_path : Callable, default=None
        Custom alignment path function to use for the distance. If None, the default
        alignment path function for the distance will be used. If the distance method
        does not have an elastic alignment path then the default SOM einsum update will
        be used. See aeon.clustering.elastic_som.VALID_ELASTIC_SOM_METRICS for a list of
        distances that have an elastic alignment path.
        The alignment path function takes the form
        Callable[[np.ndarray, np.ndarray, dict], whee the dict is the kwargs for the
        distance function. See documentation of aeon.distances documentation for
        example alignment path functions. The alignment path function must return a
        a full alignment path with no gaps.
    verbose : bool, default=False
        Verbosity mode.

    Attributes
    ----------
    cluster_centers_ : 3d np.ndarray
        Array of shape (n_clusters, n_channels, n_timepoints))
        Time series that represent each of the cluster centers.
    labels_ : 1d np.ndarray
        1d array of shape (n_case,)
        Labels that is the index each time series belongs to.

    References
    ----------
    .. [1] Vesanto, Juha & Alhoniemi, Esa. (2000). Clustering of the self-organizing
        map. IEEE Transactions on Neural Networks, 11(3), 586-600.
       https://doi.org/10.1109/72.846731.

    .. [2] Silva, Maria & Henriques, Roberto. (2020). Exploring time-series motifs
        through DTW-SOM. In Proceedings of the International Joint Conference on Neural
        Networks (IJCNN), 1-8. https://doi.org/10.1109/IJCNN48605.2020.9207614.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.clustering import ElasticSOM
    >>> X = np.random.random(size=(10,2,20))
    >>> clst = ElasticSOM(n_clusters=3, random_state=1, num_iterations=10)
    >>> clst.fit(X)
    ElasticSOM(n_clusters=3, num_iterations=10, random_state=1)
    """

    _tags = {
        "capability:multivariate": True,
        "algorithm_type": AlgorithmType.DISTANCE.value,
    }

    def __init__(
        self,
        n_clusters: int = 8,
        distance: Union[str, Callable] = "dtw",
        init: Union[str, np.ndarray] = "random",
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        decay_function: Union[Callable, str] = "asymptotic_decay",
        neighborhood_function: Union[Callable, str] = "gaussian",
        sigma_decay_function: Union[Callable, str] = "asymptotic_decay",
        num_iterations: int = 500,
        distance_params: Optional[dict] = None,
        random_state: Optional[Union[int, RandomState]] = None,
        custom_alignment_path: Optional[Callable] = None,
        verbose: Optional[bool] = False,
    ):
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.decay_function = decay_function
        self.neighborhood_function = neighborhood_function
        self.distance = distance
        self.num_iterations = num_iterations
        self.random_state = random_state
        self.distance_params = distance_params
        self.verbose = verbose
        self.init = init
        self.sigma_decay_function = sigma_decay_function
        self.custom_alignment_path = custom_alignment_path
        self.n_clusters = n_clusters

        self._random_state = None
        self._alignment_path_callable = None
        self._decay_function = None
        self._neighborhood_function = None
        self._distance_params = None
        self._init = None
        self._neuron_position = None
        self._sigma_decay_function = None

        self.labels_ = None
        self.cluster_centers_ = None
        super().__init__()

    def _fit(self, X, y=None):
        self._check_params(X)

        if isinstance(self._init, Callable):
            weights = self._init(X)
        else:
            weights = self._init.copy()

        num_iterations = self.num_iterations * X.shape[-1]
        iterations = np.arange(num_iterations) % len(X)
        self._random_state.shuffle(iterations)

        for t, iteration in enumerate(iterations):
            if self.verbose:
                print(f"Iteration {t}/{num_iterations}")  # noqa: T001, T201
            decay_rate = int(t)
            weights = self._update_iteration(
                X[iteration], weights, decay_rate, num_iterations
            )

        self.labels_ = self._find_bmu(X, weights)
        self.cluster_centers_ = weights

    def _predict(self, X, y=None):
        return self._find_bmu(X, self.cluster_centers_)

    def _find_bmu(self, x, weights):
        pairwise_matrix = pairwise_distance(
            x,
            weights,
            method=self.distance,
            **self._distance_params,
        )
        return pairwise_matrix.argmin(axis=1)

    def _update_iteration(self, x, weights, decay_rate, num_iterations):
        # Finds the best matching unit for the current time series
        bmu = self._find_bmu(x, weights)[0]
        eta = self._decay_function(self.learning_rate, decay_rate, num_iterations)
        sig = self._sigma_decay_function(self.sigma, decay_rate, num_iterations)
        g = self._neighborhood_function(self._neuron_position, bmu, sig) * eta

        # Check if an alignment path function exists for the distance
        if self._alignment_path_callable is not None:
            it = np.nditer(g, flags=["multi_index"])

            while not it.finished:
                weights[it.multi_index] = self._elastic_update(
                    x, weights[it.multi_index], g[it.multi_index]
                )
                it.iternext()
        else:
            weights += np.einsum("i, ijk->ijk", g, x - weights)

        return weights

    def _check_params(self, X):
        self._random_state = check_random_state(self.random_state)
        # random initialization
        if isinstance(self.init, str):
            if self.init == "random":
                self._init = self._random_center_initializer
            elif self.init == "kmeans++":
                self._init = self._kmeans_plus_plus_center_initializer
            elif self.init == "first":
                self._init = self._first_center_initializer
            else:
                raise ValueError(
                    f"The value provided for init: {self.init} is "
                    f"invalid. The following are a list of valid init algorithms "
                    f"strings: random, kmedoids++, first"
                )
        else:
            if isinstance(self.init, np.ndarray) and len(self.init) == self.n_clusters:
                self._init = self.init.copy()
            else:
                raise ValueError(
                    f"The value provided for init: {self.init} is "
                    f"invalid. The following are a list of valid init algorithms "
                    f"strings: random, kmedoids++, first. You can also pass a"
                    f"np.ndarray of size (n_clusters, n_channels, n_timepoints)"
                )

        self._neuron_position = np.arange(self.n_clusters)

        if self.decay_function in _lr_decay_functions:
            self._decay_function = _lr_decay_functions[self.decay_function]
        else:
            if isinstance(self.decay_function, Callable):
                self._decay_function = self.decay_function
            else:
                raise ValueError(
                    "custom_sigma_decay_function must be a callable function or in: "
                    f"{VALID_LR_DECAY_FUNCTIONS}"
                )

        if self.sigma_decay_function in _sigma_decay_functions:
            self._sigma_decay_function = _sigma_decay_functions[
                self.sigma_decay_function
            ]
        else:
            if isinstance(self.sigma_decay_function, Callable):
                self._sigma_decay_function = self.sigma_decay_function
            else:
                raise ValueError(
                    "custom_sigma_decay_function must be a callable function or in: "
                    f"{VALID_SIGMA_DECAY_FUNCTIONS}"
                )

        if self.neighborhood_function in _neighborhood_functions:
            self._neighborhood_function = _neighborhood_functions[
                self.neighborhood_function
            ]
        else:
            if isinstance(self.neighborhood_function, Callable):
                self._neighborhood_function = self.neighborhood_function
            else:
                raise ValueError(
                    f"custom_neighborhood_function must be a callable function or in: "
                    f"{VALID_NEIGHBORHOOD_FUNCTIONS}"
                )

        if isinstance(self.custom_alignment_path, Callable):
            self._alignment_path_callable = self.custom_alignment_path
        else:
            if self.distance in VALID_ELASTIC_SOM_METRICS:
                self._alignment_path_callable = get_alignment_path_function(
                    self.distance
                )
            else:
                self._alignment_path_callable = None

        if self.distance_params is None:
            self._distance_params = {}
        else:
            self._distance_params = self.distance_params

    def _elastic_update(self, x, y, w):
        best_path, distance = self._alignment_path_callable(
            x, y, **self._distance_params
        )

        x_cords = []
        y_cords = []
        for i in best_path:
            x_cords += [round(i[0] * w + i[1] * (1 - w))]
            y_cords += [x[:, i[0]] * w + y[:, i[1]] * (1 - w)]

        s3 = np.zeros_like(x)
        counts = np.zeros(x.shape[1])

        for j in range(x.shape[1]):
            indices = np.where(np.array(x_cords) == j)[0]
            if len(indices) > 0:
                s3[:, j] = np.mean([y_cords[k] for k in indices], axis=0)
                counts[j] = len(indices)

        for j in range(1, x.shape[1]):
            if counts[j] == 0:
                s3[:, j] = s3[:, j - 1]

        return s3

    def _random_center_initializer(self, X: np.ndarray) -> np.ndarray:
        return X[self._random_state.choice(X.shape[0], self.n_clusters, replace=False)]

    def _kmeans_plus_plus_center_initializer(self, X: np.ndarray):
        initial_center_idx = self._random_state.randint(X.shape[0])
        indexes = [initial_center_idx]

        for _ in range(1, self.n_clusters):
            pw_dist = pairwise_distance(
                X, X[indexes], method=self.distance, **self._distance_params
            )
            min_distances = pw_dist.min(axis=1)
            probabilities = min_distances / min_distances.sum()
            next_center_idx = self._random_state.choice(X.shape[0], p=probabilities)
            indexes.append(next_center_idx)

        centers = X[indexes]
        return centers

    def _first_center_initializer(self, X: np.ndarray) -> np.ndarray:
        return X[list(range(self.n_clusters))]

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {
            "n_clusters": 2,
            "init": "random",
            "random_state": 1,
            "num_iterations": 10,
        }


# SOM decay functions
def _asymptotic_decay(learning_rate, current_iteration, max_iter):
    return learning_rate / (1 + current_iteration / (max_iter / 2))


def _inverse_decay_to_zero(learning_rate, t, max_iter):
    c = max_iter / 100.0
    return learning_rate * c / (c + t)


def _linear_decay_to_zero(learning_rate, current_iteration, max_iter):
    return learning_rate * (1 - current_iteration / max_iter)


def _inverse_decay_to_one(sigma, current_iteration, max_iter):
    c = (sigma - 1) / max_iter
    return sigma / (1 + (current_iteration * c))


def _linear_decay_to_one(sigma, current_iteration, max_iter):
    return sigma + (current_iteration * (1 - sigma) / max_iter)


_lr_decay_functions = {
    "asymptotic_decay": _asymptotic_decay,
    "inverse_decay_to_zero": _inverse_decay_to_zero,
    "linear_decay_to_zero": _linear_decay_to_zero,
}

_sigma_decay_functions = {
    "asymptotic_decay": _asymptotic_decay,
    "inverse_decay_to_one": _inverse_decay_to_one,
    "linear_decay_to_one": _linear_decay_to_one,
}


# SOM neighborhood functions
def _gaussian_neighborhood(neuron_position, c, sigma):
    d = 2 * sigma * sigma
    return np.exp(-np.power(neuron_position - neuron_position[c], 2) / d)


def _mexican_hat(neuron_position, c, sigma):
    p = np.power(neuron_position - neuron_position[c], 2)
    d = 2 * sigma * sigma
    return np.exp(-p / d) * (1 - 2 / d * p)


_neighborhood_functions = {
    "gaussian": _gaussian_neighborhood,
    "mexican_hat": _mexican_hat,
}

VALID_LR_DECAY_FUNCTIONS = list(_lr_decay_functions.keys())
VALID_SIGMA_DECAY_FUNCTIONS = list(_sigma_decay_functions.keys())
VALID_NEIGHBORHOOD_FUNCTIONS = list(_neighborhood_functions.keys())
