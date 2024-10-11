"""Random Supervised Time Series Forest (RSTSF) Classifier."""

__maintainer__ = []
__all__ = ["RSTSF"]

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import FunctionTransformer

from aeon.classification import BaseClassifier
from aeon.transformations.collection import (
    ARCoefficientTransformer,
    PeriodogramTransformer,
)
from aeon.transformations.collection.interval_based import SupervisedIntervals
from aeon.utils.numba.general import first_order_differences_3d
from aeon.utils.validation import check_n_jobs


class RSTSF(BaseClassifier):
    """
    Random Supervised Time Series Forest (RSTSF) Classifier.

    An ensemble of decision trees built on intervals selected through a supervised
    process as described in _[1].
    Overview: Input n series of length m with d dimensions
        - sample X using class-balanced bagging
        - sample intervals for all 4 series representations and 9 features using
            supervised method
        - build extra trees classifier on transformed interval data

    Parameters
    ----------
    n_estimators : int, default=200
        The number of trees in the forest.
    n_intervals : int, default=50
        The number of times the supervised interval selection process is run.
        Each supervised extraction will output a varying amount of features based on
        series length, number of dimensions and the number of features.
    min_interval_length : int, default=3
        The minimum length of extracted intervals. Minimum value of 3.
    use_pyfftw : bool, default=True
        Whether to use pyfftw for the periodogram transformation.
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
        The number of jobs to run in parallel for both `fit` and `predict` functions.
        `-1` means using all processors.

    See Also
    --------
    SupervisedIntervals

    References
    ----------
    .. [1] Cabello, N., Naghizade, E., Qi, J. and Kulik, L., 2021. Fast, accurate and
        interpretable time series classification through randomization. arXiv preprint
        arXiv:2105.14876.

    Examples
    --------
    >>> from aeon.classification.interval_based import RSTSF
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, random_state=0)
    >>> clf = RSTSF(n_estimators=10, n_intervals=5, random_state=0)  # doctest: +SKIP
    >>> clf.fit(X, y)  # doctest: +SKIP
    RSTSF(...)
    >>> clf.predict(X)  # doctest: +SKIP
    [0 1 0 1 0 0 1 1 1 0]
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "interval",
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        n_estimators=200,
        n_intervals=50,
        min_interval_length=3,
        use_pyfftw=False,
        random_state=None,
        class_weight=None,
        n_jobs=1,
    ):
        self.n_estimators = n_estimators
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.use_pyfftw = use_pyfftw
        self.random_state = random_state
        self.class_weight = class_weight
        self.n_jobs = n_jobs

        super().__init__()

        if use_pyfftw:
            self.set_tags(**{"python_dependencies": ["statsmodels", "pyfftw"]})

    def _fit(self, X, y):
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape

        self._n_jobs = check_n_jobs(self.n_jobs)

        lags = int(12 * (X.shape[2] / 100.0) ** 0.25)

        self._series_transformers = [
            FunctionTransformer(func=first_order_differences_3d, validate=False),
            PeriodogramTransformer(use_pyfftw=self.use_pyfftw),
            ARCoefficientTransformer(order=lags, replace_nan=True),
        ]

        transforms = [X] + [t.fit_transform(X) for t in self._series_transformers]

        Xt = np.empty((X.shape[0], 0))
        self._transformers = []
        transform_data_lengths = []
        for t in transforms:
            si = SupervisedIntervals(
                n_intervals=self.n_intervals,
                min_interval_length=self.min_interval_length,
                n_jobs=self._n_jobs,
                random_state=self.random_state,
                randomised_split_point=True,
            )
            features = si.fit_transform(t, y)
            Xt = np.hstack((Xt, features))
            self._transformers.append(si)
            transform_data_lengths.append(features.shape[1])

        self.clf_ = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            criterion="entropy",
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=self._n_jobs,
            random_state=self.random_state,
        )
        self.clf_.fit(Xt, y)

        relevant_features = []
        for tree in self.clf_.estimators_:
            relevant_features.extend(tree.tree_.feature[tree.tree_.feature >= 0])
        relevant_features = np.unique(relevant_features)

        features_to_transform = [False] * Xt.shape[1]
        for i in relevant_features:
            features_to_transform[i] = True

        count = 0
        for r in range(len(transforms)):
            self._transformers[r].set_features_to_transform(
                features_to_transform[count : count + transform_data_lengths[r]],
                raise_error=False,
            )
            count += transform_data_lengths[r]

        return self

    def _predict(self, X):
        Xt = self._predict_transform(X)
        return self.clf_.predict(Xt)

    def _predict_proba(self, X):
        Xt = self._predict_transform(X)
        return self.clf_.predict_proba(Xt)

    def _predict_transform(self, X):
        transforms = [X] + [t.transform(X) for t in self._series_transformers]

        Xt = np.empty((X.shape[0], 0))
        for i, t in enumerate(transforms):
            si = self._transformers[i]
            Xt = np.hstack((Xt, si.transform(t)))

        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
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
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return {
            "n_estimators": 2,
            "n_intervals": 2,
        }
