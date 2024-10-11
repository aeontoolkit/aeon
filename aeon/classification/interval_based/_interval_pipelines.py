"""Interval Pipeline Classifiers.

Pipeline classifiers which extract interval features then build a base estimator.
"""

__maintainer__ = []
__all__ = ["RandomIntervalClassifier", "SupervisedIntervalClassifier"]

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from aeon.base._base import _clone_estimator
from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.interval_based import (
    RandomIntervals,
    SupervisedIntervals,
)


class RandomIntervalClassifier(BaseClassifier):
    """
    Random Interval Classifier.

    Extracts multiple intervals with random length, position and dimension from series
    and concatenates them into a feature vector. Builds an estimator on the
    transformed data.

    Parameters
    ----------
    n_intervals : int, default=100,
        The number of intervals of random length, position and dimension to be
        extracted.
    min_interval_length : int, default=3
        The minimum length of extracted intervals. Minimum value of 3.
    max_interval_length : int, default=3
        The maximum length of extracted intervals. Minimum value of min_interval_length.
    features : aeon transformer, a function taking a 2d numpy array parameter, or list
            of said transformers and functions, default=None
        Transformers and functions used to extract features from selected intervals.
        If None, defaults to [mean, median, min, max, std, 25% quantile, 75% quantile]
    dilation : int, list or None, default=None
        Add dilation to extracted intervals. No dilation is added if None or 1. If a
        list of ints, a random dilation value is selected from the list for each
        interval.

        The dilation value is selected after the interval star and end points. If the
        number of values in the dilated interval is less than the min_interval_length,
        the amount of dilation applied is reduced.
    estimator : sklearn classifier, optional, default=None
        An sklearn estimator to be built using the transformed data.
        Defaults to sklearn RandomForestClassifier(n_estimators=200)
    random_state : None, int or instance of RandomState, default=None
        Seed or RandomState object used for random number generation.
        If random_state is None, use the RandomState singleton used by np.random.
        If random_state is an int, use a new RandomState instance seeded with seed.
    class_weight{“balanced”, “balanced_subsample”}: dict or list of dicts, default=None
        From sklearn documentation:
        If not given, all classes are supposed to have weight one.
        The “balanced” mode uses the values of y to automatically adjust weights
        inversely proportional to class frequencies in the input data as
        n_samples / (n_classes * np.bincount(y))
        The “balanced_subsample” mode is the same as “balanced” except that weights
        are computed based on the bootstrap sample for every tree grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed through
        the fit method) if sample_weight is specified.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `transform` functions.
        `-1` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    Attributes
    ----------
    n_cases_ : int
        The number of train cases.
    n_channels_ : int
        The number of dimensions per case.
    n_timepoints_ : int
        The length of each series.
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes)
        Holds the label for each class.

    See Also
    --------
    RandomIntervals

    Examples
    --------
    >>> from aeon.classification.interval_based import RandomIntervalClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, random_state=0)
    >>> clf = RandomIntervalClassifier(
    ...     estimator=RandomForestClassifier(n_estimators=5),
    ...     n_intervals=5,
    ...     random_state=0,
    ... )
    >>> clf.fit(X, y)
    RandomIntervalClassifier(...)
    >>> clf.predict(X)
    array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "interval",
    }

    def __init__(
        self,
        n_intervals=100,
        min_interval_length=3,
        max_interval_length=np.inf,
        features=None,
        dilation=None,
        estimator=None,
        class_weight=None,
        n_jobs=1,
        random_state=None,
        parallel_backend=None,
    ):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.max_interval_length = max_interval_length
        self.features = features
        self.dilation = dilation
        self.estimator = estimator
        self.random_state = random_state
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        super().__init__()

    def _fit(self, X, y):
        """Fit RandomIntervalClassifier to training data.

        Parameters
        ----------
        X : 3D np.ndarray (any number of channels, equal length series)
                of shape (n_cases, n_channels, n_timepoints)
        y : 1D np.array, of shape [n_cases] - class labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape

        self._transformer = RandomIntervals(
            n_intervals=self.n_intervals,
            min_interval_length=self.min_interval_length,
            max_interval_length=self.max_interval_length,
            features=self.features,
            dilation=self.dilation,
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=self._n_jobs,
            parallel_backend=self.parallel_backend,
        )

        self._estimator = _clone_estimator(
            (
                RandomForestClassifier(n_estimators=200)
                if self.estimator is None
                else self.estimator
            ),
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t = self._transformer.fit_transform(X, y)
        self._estimator.fit(X_t, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray (any number of channels, equal length series)
                of shape (n_cases, n_channels, n_timepoints)

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted class labels.
        """
        return self._estimator.predict(self._transformer.transform(X))

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray (any number of channels, equal length series)
                of shape (n_cases, n_channels, n_timepoints)

        Returns
        -------
        y : array-like, shape = [n_cases, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(self._transformer.transform(X))
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(self._transformer.transform(X))
            for i in range(0, X.shape[0]):
                dists[i, self._class_dictionary[preds[i]]] = 1
            return dists

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            RandomIntervalClassifier provides the following special sets:
                 "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        from aeon.utils.numba.stats import row_mean, row_numba_min

        if parameter_set == "results_comparison":
            return {
                "n_intervals": 3,
                "estimator": RandomForestClassifier(n_estimators=10),
                "features": [row_mean, row_numba_min],
            }
        else:
            return {
                "n_intervals": 2,
                "estimator": RandomForestClassifier(n_estimators=2),
                "features": [row_mean, row_numba_min],
            }


class SupervisedIntervalClassifier(BaseClassifier):
    """Supervised Interval Classifier.

    Extracts multiple intervals from series with using a supervised process
    and concatenates them into a feature vector. Builds an estimator on the
    transformed data.

    Parameters
    ----------
    n_intervals : int, default=50
        The number of times the supervised interval selection process is run. This
        process will extract more then one interval per run.
        Each supervised extraction will output a varying amount of features based on
        series length, number of dimensions and the number of features.
    min_interval_length : int, default=3
        The minimum length of extracted intervals. Minimum value of 3.
    features : callable, list of callables, default=None
        Functions used to extract features from selected intervals. Must take a 2d
        array of shape (n_cases, interval_length) and return a 1d array of shape
        (n_cases) containing the features.
        If None, defaults to the following statistics used in [2]:
        [mean, median, std, slope, min, max, iqr, count_mean_crossing,
        count_above_mean].
    metric : ["fisher"] or callable, default="fisher"
        The metric used to evaluate the usefulness of a feature extracted on an
        interval. If "fisher", the Fisher score is used. If a callable, it must take
        a 1d array of shape (n_cases) and return a 1d array of scores of shape
        (n_cases).
    randomised_split_point : bool, default=True
        If True, the split point for interval extraction is randomised as is done in [2]
        rather than split in half.
    normalise_for_search : bool, default=True
        If True, the data is normalised for the supervised interval search process.
        Features extracted for the transform output will not use normalised data.
    estimator : sklearn classifier, optional, default=None
        An sklearn estimator to be built using the transformed data.
        Defaults to sklearn RandomForestClassifier(n_estimators=200)
    random_state : None, int or instance of RandomState, default=None
        Seed or RandomState object used for random number generation.
        If random_state is None, use the RandomState singleton used by np.random.
        If random_state is an int, use a new RandomState instance seeded with seed.
    class_weight{“balanced”, “balanced_subsample”}: dict or list of dicts, default=None
        From sklearn documentation:
        If not given, all classes are supposed to have weight one.
        The “balanced” mode uses the values of y to automatically adjust weights
        inversely proportional to class frequencies in the input data as
        n_samples / (n_classes * np.bincount(y))
        The “balanced_subsample” mode is the same as “balanced” except that weights
        are computed based on the bootstrap sample for every tree grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed through
        the fit method) if sample_weight is specified.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `transform` functions.
        `-1` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    Attributes
    ----------
    n_cases_ : int
        The number of train cases.
    n_channels_ : int
        The number of dimensions per case.
    n_timepoints_ : int
        The length of each series.
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes)
        Holds the label for each class.

    See Also
    --------
    SupervisedIntervals

    Examples
    --------
    >>> from aeon.classification.interval_based import SupervisedIntervalClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, random_state=0)
    >>> clf = SupervisedIntervalClassifier(
    ...     estimator=RandomForestClassifier(n_estimators=5),
    ...     n_intervals=2,
    ...     random_state=0,
    ... )
    >>> clf.fit(X, y)
    SupervisedIntervalClassifier(...)
    >>> clf.predict(X)
    array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "interval",
    }

    def __init__(
        self,
        n_intervals=50,
        min_interval_length=3,
        features=None,
        metric="fisher",
        randomised_split_point=True,
        normalise_for_search=True,
        estimator=None,
        random_state=None,
        class_weight=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.features = features
        self.metric = metric
        self.randomised_split_point = randomised_split_point
        self.normalise_for_search = normalise_for_search
        self.estimator = estimator
        self.random_state = random_state
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        super().__init__()

    def _fit(self, X, y):
        """Fit SupervisedIntervalClassifier to training data.

        Parameters
        ----------
        X : 3D np.ndarray (any number of channels, equal length series)
                of shape (n_cases, n_channels, n_timepoints)
        y : 1D np.array, of shape [n_cases] - class labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape

        self._transformer = SupervisedIntervals(
            n_intervals=self.n_intervals,
            min_interval_length=self.min_interval_length,
            features=self.features,
            metric=self.metric,
            randomised_split_point=self.randomised_split_point,
            normalise_for_search=self.normalise_for_search,
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            parallel_backend=self.parallel_backend,
        )

        self._estimator = _clone_estimator(
            (
                RandomForestClassifier(n_estimators=200)
                if self.estimator is None
                else self.estimator
            ),
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t = self._transformer.fit_transform(X, y)
        self._estimator.fit(X_t, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray (any number of channels, equal length series)
                of shape (n_cases, n_channels, n_timepoints)

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted class labels.
        """
        return self._estimator.predict(self._transformer.transform(X))

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray (any number of channels, equal length series)
                of shape (n_cases, n_channels, n_timepoints)

        Returns
        -------
        y : array-like, shape = [n_cases, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(self._transformer.transform(X))
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(self._transformer.transform(X))
            for i in range(0, X.shape[0]):
                dists[i, self._class_dictionary[preds[i]]] = 1
            return dists

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            SupervisedIntervalClassifier provides the following special sets:
                 "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        from aeon.utils.numba.stats import row_mean, row_numba_min

        if parameter_set == "results_comparison":
            return {
                "n_intervals": 2,
                "estimator": RandomForestClassifier(n_estimators=10),
                "features": [row_mean, row_numba_min],
            }
        else:
            return {
                "n_intervals": 1,
                "estimator": RandomForestClassifier(n_estimators=2),
                "features": [row_mean, row_numba_min],
            }
