"""TSFresh Classifier.

Pipeline classifier using the TSFresh transformer and an estimator.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["TSFreshClassifier"]

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from aeon.base._base import _clone_estimator
from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.feature_based import TSFresh, TSFreshRelevant


class TSFreshClassifier(BaseClassifier):
    """
    Time Series Feature Extraction based on Scalable Hypothesis Tests classifier.

    This classifier simply transforms the input data using the TSFresh [1]_
    transformer and builds a provided estimator using the transformed data.

    Parameters
    ----------
    default_fc_parameters : str, default="efficient"
        Set of TSFresh features to be extracted, options are "minimal", "efficient" or
        "comprehensive".
    relevant_feature_extractor : bool, default=False
        Remove irrelevant features using the FRESH algorithm.
    estimator : sklearn classifier, default=None
        An sklearn estimator to be built using the transformed data. Defaults to a
        Random Forest with 200 trees.
    verbose : int, default=0
        Level of output printed to the console (for information only).
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    chunksize : int or None, default=None
        Number of series processed in each parallel TSFresh job, should be optimised
        for efficient parallelisation.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    class_weight{“balanced”, “balanced_subsample”}, dict or list of dicts, default=None
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

    Attributes
    ----------
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.
    estimator_ : sklearn classifier
        The fitted estimator.

    See Also
    --------
    TSFresh
    TSFreshRelevant
    TSFreshRegressor

    References
    ----------
    .. [1] Christ, Maximilian, et al. "Time series feature extraction on basis of
        scalable hypothesis tests (tsfresh–a python package)." Neurocomputing 307
        (2018): 72-77.
        https://www.sciencedirect.com/science/article/pii/S0925231218304843
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "feature",
        "python_dependencies": "tsfresh",
    }

    def __init__(
        self,
        default_fc_parameters="efficient",
        relevant_feature_extractor=True,
        estimator=None,
        verbose=0,
        n_jobs=1,
        chunksize=None,
        random_state=None,
        class_weight=None,
    ):
        self.default_fc_parameters = default_fc_parameters
        self.relevant_feature_extractor = relevant_feature_extractor
        self.estimator = estimator

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.random_state = random_state

        self._transformer = None
        self._return_majority_class = False
        self._majority_class = 0
        self.class_weight = class_weight

        super().__init__()

    def _fit(self, X, y):
        """Fit a pipeline on cases (X,y), where y is the target variable.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The training data.
        y : array-like, shape = [n_cases]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        self._transformer = (
            TSFreshRelevant(
                default_fc_parameters=self.default_fc_parameters,
                n_jobs=self._n_jobs,
                chunksize=self.chunksize,
            )
            if self.relevant_feature_extractor
            else TSFresh(
                default_fc_parameters=self.default_fc_parameters,
                n_jobs=self._n_jobs,
                chunksize=self.chunksize,
            )
        )
        self.estimator_ = _clone_estimator(
            (
                RandomForestClassifier(n_estimators=200, class_weight=self.class_weight)
                if self.estimator is None
                else self.estimator
            ),
            self.random_state,
        )

        if self.verbose < 2:
            self._transformer.show_warnings = False
            if self.verbose < 1:
                self._transformer.disable_progressbar = True

        m = getattr(self.estimator_, "n_jobs", None)
        if m is not None:
            self.estimator_.n_jobs = self._n_jobs

        X_t = self._transformer.fit_transform(X, y)

        if X_t.shape[1] == 0:
            warnings.warn(
                "TSFresh has extracted no features from the data. Returning the "
                "majority class in predictions. Setting "
                "relevant_feature_extractor=False will keep all features.",
                UserWarning,
                stacklevel=2,
            )

            self._return_majority_class = True
            self._majority_class = np.argmax(np.unique(y, return_counts=True)[1])
        else:
            self.estimator_.fit(X_t, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted class labels.
        """
        if self._return_majority_class:
            return np.full(X.shape[0], self.classes_[self._majority_class])

        return self.estimator_.predict(self._transformer.transform(X))

    def _predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_cases, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        if self._return_majority_class:
            dists = np.zeros((X.shape[0], self.n_classes_))
            dists[:, self._majority_class] = 1
            return dists

        m = getattr(self.estimator_, "predict_proba", None)
        if callable(m):
            return self.estimator_.predict_proba(self._transformer.transform(X))
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self.estimator_.predict(self._transformer.transform(X))
            for i in range(0, X.shape[0]):
                dists[i, self._class_dictionary[preds[i]]] = 1
            return dists

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            TSFreshClassifier provides the following special sets:
                 "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        if parameter_set == "results_comparison":
            return {
                "estimator": RandomForestClassifier(n_estimators=10),
                "default_fc_parameters": "minimal",
                "relevant_feature_extractor": False,
            }
        else:
            return {
                "estimator": RandomForestClassifier(n_estimators=2),
                "default_fc_parameters": "minimal",
                "relevant_feature_extractor": False,
            }
