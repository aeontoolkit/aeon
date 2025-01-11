"""Proximity Forest 2.0 Classifier."""

from typing import Any, Callable, Optional, TypedDict, Union

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from numba.typed import List as NumbaList
from sklearn.utils import check_random_state
from typing_extensions import Unpack

from aeon.classification.base import BaseClassifier
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic._lcss import lcss_distance
from aeon.distances.pointwise._minkowski import (
    _univariate_minkowski_distance,
    minkowski_distance,
)


class ProximityForest2(BaseClassifier):
    """Proximity Forest 2.0 Classifier.

    The Proximity Forest 2.0 outperforms the Proximity Forest Classifier.
    PF 2.0 incorporates three recent advances in time series similarity measures
    (1) computationally e cient early abandoning and pruning to speedup elastic
    similarity computations; (2) a new elastic similarity measure, Amerced Dynamic
    Time Warping (ADTW); and (3) cost function tuning.

    Parameters
    ----------
    n_trees: int, default = 100
        The number of trees, by default an ensemble of 100 trees is formed.
    n_splitters: int, default = 5
        The number of candidate splitters to be evaluated at each node.
    max_depth: int, default = None
        The maximum depth of the tree. If None, then nodes are expanded until all
        leaves are pure or until all leaves contain less than min_samples_split samples.
    min_samples_split: int, default = 2
        The minimum number of samples required to split an internal node.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default = 1
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details. Parameter for compatibility purposes, still unimplemented.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    Notes
    -----
    For the C++ version, see
    `ProximityForest2.0
    <https://github.com/MonashTS/ProximityForest2.0>`_.

    Also refer to the Proximity Forest implementation,
    see the :class:`~aeon.classification.distance_based.ProximityForest` API.

    References
    ----------
    .. [1] Matthieu Herrmann, Chang Wei Tan, Mahsa Salehi, Geoffrey I. Webb.
    Proximity Forest 2.0: A new e ective and scalable similarity-based classifier
    for time series, https://doi.org/10.48550/arXiv.2304.05800.

    Examples
    --------
    >>> from aeon.datasets import load_unit_test
    >>> from aeon.classification.distance_based import ProximityForest
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> classifier = ProximityForest2(n_trees = 10, n_splitters = 3)
    >>> classifier.fit(X_train, y_train)
    ProximityForest2(...)
    >>> y_pred = classifier.predict(X_test)
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:multithreading": True,
        "algorithm_type": "distance",
        "X_inner_type": ["numpy2D"],
    }

    def __init__(
        self,
        n_trees=100,
        n_splitters: int = 5,
        max_depth: int = None,
        min_samples_split: int = 2,
        random_state: Union[int, type[np.random.RandomState], None] = None,
        n_jobs: int = 1,
        parallel_backend=None,
    ):
        self.n_trees = n_trees
        self.n_splitters = n_splitters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        super().__init__()

    def _fit(self, X, y):
        rng = check_random_state(self.random_state)
        seeds = rng.randint(np.iinfo(np.int32).max, size=self.n_trees)
        self.trees_ = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(_fit_tree)(
                X,
                y,
                self.n_splitters,
                self.max_depth,
                self.min_samples_split,
                check_random_state(seed),
            )
            for seed in seeds
        )

    def _predict_proba(self, X):
        classes = list(self.classes_)
        preds = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(delayed(_predict_tree)(tree, X) for tree in self.trees_)
        n_cases = X.shape[0]
        votes = np.zeros((n_cases, self.n_classes_))
        for i in range(len(preds)):
            predictions = np.array(
                [classes.index(class_label) for class_label in preds[i]]
            )
            for j in range(n_cases):
                votes[j, predictions[j]] += 1
        output_probas = votes / self.n_trees
        return output_probas

    def _predict(self, X):
        probas = self._predict_proba(X)
        idx = np.argmax(probas, axis=1)
        preds = np.asarray([self.classes_[x] for x in idx])
        return preds


def _fit_tree(X, y, n_splitters, max_depth, min_samples_split, random_state):
    clf = ProximityTree2(
        n_splitters=n_splitters,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf


def _predict_tree(tree, X):
    return tree.predict(X)


class _Node:
    """Proximity Tree node.

    Parameters
    ----------
    node_id: str
        The id of node, root node has id 0.
    _is_leaf: bool
        To identify leaf nodes.
    label: int, str or None
        Contains the class label of leaf node, None otherwise.
    splitter: dict
        The splitter used to split the node.
    class_distribution: dict
        In case of unpure leaf node, save the class distribution to calculate
        probability of each class.
    children: dict
        Contains the class label and the associated node, empty for leaf node.
    """

    def __init__(
        self,
        node_id: str,
        _is_leaf: bool,
        label=None,
        class_distribution=None,
        splitter=None,
    ):
        self.node_id = node_id
        self._is_leaf = _is_leaf
        self.label = label
        self.splitter = splitter
        self.class_distribution = class_distribution or {}
        self.children = {}


class ProximityTree2(BaseClassifier):
    """Proximity Tree classifier for PF2.0.

    It leverages ADTW and tuning the cost functions for TSC to increase accuracy,
    together with the computational e ciency of computing these distances provided by
    EAP.

    PF 2.0 modifies the splitters to comprise three elements, a set of class exemplars,
    a parameterised similarity measure and a time series transform.

    Parameters
    ----------
    n_splitters: int, default = 5
        The number of candidate splitters to be evaluated at each node.
    max_depth: int, default = None
        The maximum depth of the tree. If None, then nodes are expanded until all
        leaves are pure or until all leaves contain less than min_samples_split samples.
    min_samples_split: int, default = 2
        The minimum number of samples required to split an internal node.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Notes
    -----
    For the C++ version, see
    `ProximityForest2.0
    <https://github.com/MonashTS/ProximityForest2.0>`_.

    Also refer to the Proximity Tree implementation,
    see the :class:`~aeon.classification.distance_based.ProximityTree` API.

    References
    ----------
    .. [1] Matthieu Herrmann, Chang Wei Tan, Mahsa Salehi, Geoffrey I. Webb.
    Proximity Forest 2.0: A new e ective and scalable similarity-based classifier
    for time series, https://doi.org/10.48550/arXiv.2304.05800.

    Examples
    --------
    >>> from aeon.datasets import load_unit_test
    >>> from aeon.classification.distance_based import ProximityTree
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> classifier = ProximityTree2(n_splitters = 3)
    >>> classifier.fit(X_train, y_train)
    ProximityTree2(...)
    >>> y_pred = classifier.predict(X_test)
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "algorithm_type": "distance",
        "X_inner_type": ["numpy2D"],
    }

    def __init__(
        self,
        n_splitters: int = 5,
        max_depth: int = None,
        min_samples_split: int = 2,
        random_state: Union[int, type[np.random.RandomState], None] = None,
    ) -> None:
        self.n_splitters = n_splitters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        super().__init__()

    def _get_parameter_value(self, X):
        """Generate random parameter values.

        For a list of distance measures, generate a dictionary
        of parameterized distances.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_timepoints)

        Returns
        -------
        distance_param : a dictionary of distances and their
        parameters.
        """
        rng = check_random_state(self.random_state)

        X_std = X.std()
        param_ranges = {
            "dtw": {"window": (0, 0.25)},
            "adtw": {"window": (0, 0.25)},
            "lcss": {"epsilon": (X_std / 5, X_std), "window": (0, 0.25)},
        }
        random_params = {}
        for measure, ranges in param_ranges.items():
            random_params[measure] = {
                param: np.round(rng.uniform(low, high), 3)
                for param, (low, high) in ranges.items()
            }

        return random_params

    def _get_candidate_splitter(self, X, y):
        """Generate candidate splitter.

        Takes a time series dataset and a set of parameterized
        distance measures to create a candidate splitter, which
        contains a parameterized distance measure and a set of exemplars.

        Parameters
        ----------
        X : np.ndarray shape (n_cases, n_timepoints)
            The training input samples.
        y : np.array shape (n_cases,) or (n_cases,1)
            The labels of the training samples.
        parameterized_distances : dictionary
            Contains the distances and their parameters.

        Returns
        -------
        splitter : list of two dictionaries
            A distance and its parameter values and a set of exemplars.
        """
        rng = check_random_state(self.random_state)

        # Class exemplars
        exemplars = {}
        for label in np.unique(y):
            y_new = y[y == label]
            X_new = X[y == label]
            id = rng.randint(0, X_new.shape[0])
            exemplars[y_new[id]] = X_new[id, :]

        # Time series transform
        transforms = ["raw", "first_derivative"]
        t = rng.randint(0, 2)
        transform = transforms[t]

        # random parameterized distance measure
        parameterized_distances = self._get_parameter_value(X)
        n = rng.randint(0, 3)
        dist = list(parameterized_distances.keys())[n]
        if dist == "dtw":
            values = [0.5, 1, 2]
            p = rng.choice(values)
            parameterized_distances[dist]["p"] = p
        if dist == "adtw":
            values = [0.5, 1, 2]
            p = rng.choice(values)
            parameterized_distances[dist]["p"] = p
            i = rng.randint(1, 101)
            w = ((i / 100) ** 5) * self._max_warp_penalty
            parameterized_distances[dist]["warp_penalty"] = w

        # Create a list of class exemplars, distance measures and transform
        splitter = [exemplars, {dist: parameterized_distances[dist]}, transform]

        return splitter

    def _get_best_splitter(self, X, y):
        """Get the splitter for a node which maximizes the gini gain."""
        max_gain = float("-inf")
        best_splitter = None
        for _ in range(self.n_splitters):
            splitter = self._get_candidate_splitter(X, y)
            labels = list(splitter[0].keys())
            measure = list(splitter[1].keys())[0]
            transform = splitter[2]
            if transform == "first_derivative":
                X_trans = NumbaList()
                for i in range(len(X)):
                    X_trans.append(first_order_derivative(X[i]))
                exemplars = NumbaList()
                for i in range(len(labels)):
                    exemplars.append(
                        first_order_derivative(list(splitter[0].values())[i])
                    )
            else:
                X_trans = X
                exemplars = np.array(list(splitter[0].values()))

            y_subs = [[] for _ in range(len(labels))]
            for j in range(len(X_trans)):
                min_dist = float("inf")
                sub = None
                for k in range(len(labels)):
                    if measure == "dtw" or measure == "adtw":
                        splitter[1][measure]["threshold"] = min_dist
                    dist = distance(
                        X_trans[j],
                        exemplars[k],
                        metric=measure,
                        **splitter[1][measure],
                    )
                    if dist < min_dist:
                        min_dist = dist
                        sub = k
                y_subs[sub].append(y[j])
            y_subs = [np.array(ele, dtype=y.dtype) for ele in y_subs]
            gini_index = gini_gain(y, y_subs)
            if gini_index > max_gain:
                max_gain = gini_index
                best_splitter = splitter
        return best_splitter

    def _build_tree(self, X, y, depth, node_id, parent_target_value=None):
        """Build the tree recursively from the root node down to the leaf nodes."""
        # If the data reaching the node is empty
        if len(X) == 0:
            leaf_label = parent_target_value
            leaf_distribution = {}
            leaf = _Node(
                node_id=node_id,
                _is_leaf=True,
                label=leaf_label,
                class_distribution=leaf_distribution,
            )
            return leaf

        # Target value in current node
        target_value = self._find_target_value(y)
        class_distribution = {
            label: count / len(y)
            for label, count in zip(*np.unique(y, return_counts=True))
        }

        # Pure node
        if len(np.unique(y)) == 1:
            leaf_label = target_value
            leaf = _Node(
                node_id=node_id,
                _is_leaf=True,
                label=leaf_label,
                class_distribution=class_distribution,
            )
            return leaf

        # If min sample splits is reached
        if self.min_samples_split >= len(X):
            leaf_label = target_value
            leaf = _Node(
                node_id=node_id,
                _is_leaf=True,
                label=leaf_label,
                class_distribution=class_distribution,
            )
            return leaf

        # If max depth is reached
        if (self.max_depth is not None) and (depth >= self.max_depth):
            leaf_label = target_value
            leaf = _Node(
                node_id=node_id,
                _is_leaf=True,
                label=leaf_label,
                class_distribution=class_distribution,
            )
            return leaf

        # Find the best splitter
        splitter = self._get_best_splitter(X, y)

        # Create root node
        node = _Node(node_id=node_id, _is_leaf=False, splitter=splitter)

        # For each exemplar split the data
        labels = list(splitter[0].keys())
        measure = list(splitter[1].keys())[0]
        transform = splitter[2]
        if transform == "first_derivative":
            X_trans = NumbaList()
            for i in range(len(X)):
                X_trans.append(first_order_derivative(X[i]))
            exemplars = NumbaList()
            for i in range(len(labels)):
                exemplars.append(first_order_derivative(list(splitter[0].values())[i]))
        else:
            X_trans = X
            exemplars = np.array(list(splitter[0].values()))

        X_child = [[] for _ in labels]
        y_child = [[] for _ in labels]
        for i in range(len(X_trans)):
            min_dist = np.inf
            id = None
            for j in range(len(labels)):
                if measure == "dtw" or measure == "adtw":
                    splitter[1][measure]["threshold"] = min_dist
                dist = distance(
                    X_trans[i],
                    exemplars[j],
                    metric=measure,
                    **splitter[1][measure],
                )
                if dist < min_dist:
                    min_dist = dist
                    id = j
            X_child[id].append(X[i])
            y_child[id].append(y[i])
        X_child = [np.array(ele) for ele in X_child]
        y_child = [np.array(ele) for ele in y_child]
        # For each exemplar, create a branch
        for i in range(len(labels)):
            child_node_id = node_id + "." + str(i)
            child_node = self._build_tree(
                X_child[i],
                y_child[i],
                depth=depth + 1,
                node_id=child_node_id,
                parent_target_value=target_value,
            )
            node.children[labels[i]] = child_node

        return node

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _find_target_value(y):
        """Get the class label of highest frequency."""
        unique_labels = list(np.unique(y))
        class_counts = []
        for i in range(len(unique_labels)):
            cnt = 0
            for j in range(len(y)):
                if y[j] == unique_labels[i]:
                    cnt += 1
            class_counts.append(cnt)
        class_counts = np.array(class_counts)
        # Find the index of the maximum count
        max_index = np.argmax(class_counts)
        mode_value = unique_labels[max_index]
        # mode_count = counts[max_index]
        return mode_value

    def _fit(self, X, y):
        self._max_warp_penalty = _global_warp_penalty(X)
        self.root = self._build_tree(
            X, y, depth=0, node_id="0", parent_target_value=None
        )

    def _predict(self, X):
        probas = self._predict_proba(X)
        predictions = np.argmax(probas, axis=1)
        return np.array([self.classes_[pred] for pred in predictions])

    def _predict_proba(self, X):
        # Get the unique class labels
        classes = list(self.classes_)
        class_count = len(classes)
        probas = []

        for i in range(len(X)):
            # Classify the data point and find the leaf node
            leaf_node = self._classify(self.root, X[i])

            # Create probability distribution based on class counts in the leaf node
            proba = np.zeros(class_count)
            for class_label, class_proba in leaf_node.class_distribution.items():
                proba[classes.index(class_label)] = class_proba
            probas.append(proba)

        return np.array(probas)

    def _classify(self, treenode, x):
        # Classify one data point using the proximity tree
        if treenode._is_leaf:
            return treenode
        else:
            measure = list(treenode.splitter[1].keys())[0]
            branches = list(treenode.splitter[0].keys())
            transform = treenode.splitter[2]
            if transform == "first_derivative":
                x_trans = first_order_derivative(x)
                exemplars = NumbaList()
                for i in range(len(list(treenode.splitter[0].values()))):
                    exemplars.append(
                        first_order_derivative(list(treenode.splitter[0].values())[i])
                    )
            else:
                x_trans = x
                exemplars = np.array(list(treenode.splitter[0].values()))
            min_dist = np.inf
            id = None
            for i in range(len(branches)):
                if measure == "dtw" or measure == "adtw":
                    treenode.splitter[1][measure]["threshold"] = min_dist
                dist = distance(
                    x_trans,
                    exemplars[i],
                    metric=measure,
                    **treenode.splitter[1][measure],
                )
                if dist < min_dist:
                    min_dist = dist
                    id = i
            return self._classify(treenode.children[branches[id]], x)


@njit(cache=True, fastmath=True)
def gini(y) -> float:
    """Get gini score at a specific node.

    Parameters
    ----------
    y : 1d numpy array
        array of class labels

    Returns
    -------
    score : float
        gini score for the set of class labels (i.e. how pure they are). A
        larger score means more impurity. Zero means
        pure.
    """
    # get number instances at node
    n_instances = y.shape[0]
    if n_instances > 0:
        # count each class
        unique_labels = list(np.unique(y))
        class_counts = []
        for i in range(len(unique_labels)):
            cnt = 0
            for j in range(len(y)):
                if y[j] == unique_labels[i]:
                    cnt += 1
            class_counts.append(cnt)
        class_counts = np.array(class_counts)
        # subtract class entropy from current score for each class
        class_counts = np.divide(class_counts, n_instances)
        class_counts = np.power(class_counts, 2)
        sum = np.sum(class_counts)
        return 1 - sum
    else:
        # y is empty, therefore considered pure
        raise ValueError("y empty")


@njit(cache=True, fastmath=True)
def gini_gain(y, y_subs) -> float:
    """Get gini score of a split, i.e. the gain from parent to children.

    Parameters
    ----------
    y : 1d array
        array of class labels at parent
    y_subs : list of 1d array like
        list of array of class labels, one array per child

    Returns
    -------
    score : float
        gini score of the split from parent class labels to children. Note a
        higher score means better gain,
        i.e. a better split
    """
    # find number of instances overall
    parent_n_instances = y.shape[0]
    # if parent has no instances then is pure
    if parent_n_instances == 0:
        for child in y_subs:
            if len(child) > 0:
                raise ValueError("children populated but parent empty")
        return 0.5
    # find gini for parent node
    score = gini(y)
    # sum the children's gini scores
    for index in range(len(y_subs)):
        child_class_labels = y_subs[index]
        # ignore empty children
        if len(child_class_labels) > 0:
            # find gini score for this child
            child_score = gini(child_class_labels)
            # weight score by proportion of instances at child compared to
            # parent
            child_size = len(child_class_labels)
            child_score *= child_size / parent_n_instances
            # add to cumulative sum
            score -= child_score
    return score


@njit(cache=True, fastmath=True)
def _global_warp_penalty(X):
    n = len(X)
    pairs = [
        (X[np.random.randint(0, n)], X[np.random.randint(0, n)]) for _ in range(4000)
    ]
    distances = np.array([minkowski_distance(pair[0], pair[1], 2) for pair in pairs])
    penalty = np.mean(distances)
    return penalty


@njit(cache=True, fastmath=True)
def first_order_derivative_1d(q: np.ndarray) -> np.ndarray:
    """Compute the first-order derivative of a 1D time series."""
    n = len(q)
    result = np.zeros(n)

    # First element
    result[0] = (q[1] - q[0]) / 2.0

    # Middle elements
    for j in range(1, n - 1):
        result[j] = ((q[j] - q[j - 1]) + (q[j + 1] - q[j - 1]) / 2.0) / 2.0

    # Last element
    result[-1] = (q[-1] - q[-2]) / 2.0

    return result


@njit(cache=True, fastmath=True)
def first_order_derivative_2d(q: np.ndarray) -> np.ndarray:
    """Compute the first-order derivative of a 2D time series."""
    n = q.shape[1]
    result = np.zeros((1, n))

    # First element
    result[0, 0] = (q[0, 1] - q[0, 0]) / 2.0

    # Middle elements
    for j in range(1, n - 1):
        result[0, j] = (
            (q[0, j] - q[0, j - 1]) + (q[0, j + 1] - q[0, j - 1]) / 2.0
        ) / 2.0

    # Last element
    result[0, -1] = (q[0, -1] - q[0, -2]) / 2.0

    return result


@njit(cache=True, fastmath=True)
def first_order_derivative(q: np.ndarray) -> np.ndarray:
    """
    Compute the first-order derivative of the time series.

    Parameters
    ----------
    q : np.ndarray (n_timepoints,) or (1, n_timepoints)
        Time series to take derivative of.

    Returns
    -------
    np.ndarray
        The first-order derivative of q with the same shape as the input.
    """
    if q.ndim == 1:
        return first_order_derivative_1d(q)
    elif q.ndim == 2 and q.shape[0] == 1:
        return first_order_derivative_2d(q)
    else:
        raise ValueError(
            "Time series must be either (n_timepoints,) or (1, n_timepoints)."
        )


# Distances for PF 2.


class DistanceKwargs(TypedDict, total=False):
    window: Optional[float]
    itakura_max_slope: Optional[float]
    p: float
    w: np.ndarray
    epsilon: float
    warp_penalty: float
    threshold: float


DistanceFunction = Callable[[np.ndarray, np.ndarray, Any], float]


def distance(
    x: np.ndarray,
    y: np.ndarray,
    metric: Union[str, DistanceFunction],
    **kwargs: Unpack[DistanceKwargs],
) -> float:
    r"""Compute the distance between two time series.

    Sourced from the distance module to use in the PF 2.0 algorithm.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    metric : str or Callable
        The distance metric to use.
        A list of valid distance metrics can be found in the documentation for
        :func:`aeon.distances.get_distance_function` or by calling  the function
        :func:`aeon.distances.get_distance_function_names`.
    kwargs : Any
        Arguments for metric. Refer to each metrics documentation for a list of
        possible arguments.

    Returns
    -------
    float
        Distance between the x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, or 2D arrays.
        If metric is not a valid string or callable.
    """
    if metric == "minkowski":
        return minkowski_distance(x, y, kwargs.get("p", 2.0), kwargs.get("w", None))
    elif metric == "dtw":
        return _dtw_distance(
            x,
            y,
            p=kwargs.get("p"),
            window=kwargs.get("window"),
            itakura_max_slope=kwargs.get("itakura_max_slope"),
            threshold=kwargs.get("threshold"),
        )
    elif metric == "lcss":
        return lcss_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("epsilon", 1.0),
            kwargs.get("itakura_max_slope"),
        )
    elif metric == "adtw":
        return _adtw_distance(
            x,
            y,
            p=kwargs.get("p"),
            itakura_max_slope=kwargs.get("itakura_max_slope"),
            window=kwargs.get("window"),
            warp_penalty=kwargs.get("warp_penalty", 1.0),
            threshold=kwargs.get("threshold"),
        )
    else:
        if isinstance(metric, Callable):
            return metric(x, y, **kwargs)
        raise ValueError("Metric must be one of the supported strings or a callable")


@njit(cache=True, fastmath=True)
def _dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    p: float = 2.0,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    threshold: Optional[float] = np.inf,
) -> float:
    r"""Return parameterised DTW distance for PF 2.0.

    DTW is the most widely researched and used elastic distance measure. It mitigates
    distortions in the time axis by realligning (warping) the series to best match
    each other. A good background into DTW can be found in [1]_. For two series,
    possibly of unequal length,
    :math:`\\mathbf{x}=\\{x_1,x_2,\\ldots,x_n\\}` and
    :math:`\\mathbf{y}=\\{y_1,y_2, \\ldots,y_m\\}` DTW first calculates
    :math:`M(\\mathbf{x},\\mathbf{y})`, the :math:`n \times m`
    pointwise distance matrix between series :math:`\\mathbf{x}` and :math:`\\mathbf{y}`
    , where :math:`M_{i,j}=   (x_i-y_j)^p`.
    A warping path
    .. math::
        P = <(e_1, f_1), (e_2, f_2), \\ldots, (e_s, f_s)>
    is a set of pairs of indices that  define a traversal of matrix :math:`M`. A
    valid warping path must start at location :math:`(1,1)` and end at point :math:`(
    n,m)` and not backtrack, i.e. :math:`0 \\leq e_{i+1}-e_{i} \\leq 1` and :math:`0
    \\leq f_{i+1}- f_i \\leq 1` for all :math:`1< i < m`.
    The DTW distance between series is the path through :math:`M` that minimizes the
    total distance. The distance for any path :math:`P` of length :math:`s` is
    .. math::
        D_P(\\mathbf{x},\\mathbf{y}, M) =\\sum_{i=1}^s M_{e_i,f_i}
    If :math:`\\mathcal{P}` is the space of all possible paths, the DTW path :math:`P^*`
    is the path that has the minimum distance, hence the DTW distance between series is
    .. math::
        d_{dtw}(\\mathbf{x}, \\mathbf{x}) =D_{P*}(\\mathbf{x},\\mathbf{x}, M).
    The optimal warping path :math:`P^*` can be found exactly through a dynamic
    programming formulation. This can be a time consuming operation, and it is common to
    put a restriction on the amount of warping allowed. This is implemented through
    the bounding_matrix structure, that supplies a mask for allowable warpings.
    The most common bounding strategies include the Sakoe-Chiba band [2]_. The width
    of the allowed warping is controlled through the ``window`` parameter
    which sets the maximum proportion of warping allowed.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    p : float, default=2.0
        The order of the norm of the difference
        (default is 2.0, which represents the Euclidean distance).
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.
    threshold : float, default=np.inf
        Threshold to stop the calculation of cost matrix.

    Returns
    -------
    float
        DTW distance between x and y, minimum value 0.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    .. [1] Ratanamahatana C and Keogh E.: Three myths about dynamic time warping data
    mining, Proceedings of 5th SIAM International Conference on Data Mining, 2005.
    .. [2] Sakoe H. and Chiba S.: Dynamic programming algorithm optimization for
    spoken word recognition. IEEE Transactions on Acoustics, Speech, and Signal
    Processing 26(1):43–49, 1978.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _dtw_cost_matrix(_x, _y, p, bounding_matrix, threshold)[
            _x.shape[1] - 1, _y.shape[1] - 1
        ]
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _dtw_cost_matrix(x, y, p, bounding_matrix, threshold)[
            x.shape[1] - 1, y.shape[1] - 1
        ]
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _dtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    p: float,
    bounding_matrix: np.ndarray,
    threshold: float,
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0
    _w = np.ones_like(x)
    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                cost_matrix[i + 1, j + 1] = _univariate_minkowski_distance(
                    x[:, i], y[:, j], p, _w[:, i]
                ) + min(
                    cost_matrix[i, j + 1],
                    cost_matrix[i + 1, j],
                    cost_matrix[i, j],
                )
                if cost_matrix[i + 1, j + 1] > threshold:
                    break
    return cost_matrix[1:, 1:]


@njit(cache=True, fastmath=True)
def _adtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    p: float = 2.0,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    warp_penalty: float = 1.0,
    threshold: Optional[float] = np.inf,
) -> float:
    """Parameterised version of ADTW distance for PF 2.0.

    Amercing Dynamic Time Warping (ADTW) [1]_ is a variant of DTW that uses a
    explicit warping penalty to encourage or discourage warping. The warping
    penalty is a constant value that is added to the cost of warping. A high
    value will encourage the algorithm to warp less and if the value is low warping
    is more likely.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    p : float, default=2.0
        The order of the norm of the difference
        (default is 2.0, which represents the Euclidean distance).
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0.0 and 1.0
    warp_penalty: float, default=1.0
        Penalty for warping. A high value will mean less warping.
    threshold: float, default=np.inf
        The threshold to stop the calculation of cost matrix.

    Returns
    -------
    float
        ADTW distance between x and y, minimum value 0.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    .. [1] Matthieu Herrmann, Geoffrey I. Webb: Amercing: An intuitive and effective
    constraint for dynamic time warping, Pattern Recognition, Volume 137, 2023.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _adtw_cost_matrix(_x, _y, p, bounding_matrix, warp_penalty, threshold)[
            _x.shape[1] - 1, _y.shape[1] - 1
        ]
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _adtw_cost_matrix(x, y, p, bounding_matrix, warp_penalty, threshold)[
            x.shape[1] - 1, y.shape[1] - 1
        ]
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _adtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    p: float,
    bounding_matrix: np.ndarray,
    warp_penalty: float,
    threshold: float,
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    _w = np.ones_like(x)
    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                cost_matrix[i + 1, j + 1] = _univariate_minkowski_distance(
                    x[:, i], y[:, j], p, _w[:, i]
                ) + min(
                    cost_matrix[i, j + 1] + warp_penalty,
                    cost_matrix[i + 1, j] + warp_penalty,
                    cost_matrix[i, j],
                )
                if cost_matrix[i + 1, j + 1] > threshold:
                    break

    return cost_matrix[1:, 1:]
