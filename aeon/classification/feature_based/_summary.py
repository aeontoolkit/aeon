"""Summary Classifier.

Pipeline classifier using the basic summary statistics and an estimator.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["SummaryClassifier"]

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from aeon.base._base import _clone_estimator
from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.feature_based import SevenNumberSummary


class SummaryClassifier(BaseClassifier):
    """
    Summary statistic classifier.

    This classifier simply transforms the input data using the
    SevenNumberSummary transformer and builds a provided estimator using the
    transformed data.

    Parameters
    ----------
    summary_stats : ["default", "percentiles", "bowley", "tukey"], default="default"
        The summary statistics to compute.
        The options are as follows, with float denoting the percentile value extracted
        from the series:
            - "default": mean, std, min, max, 0.25, 0.5, 0.75
            - "percentiles": 0.215, 0.887, 0.25, 0.5, 0.75, 0.9113, 0.9785
            - "bowley": min, max, 0.1, 0.25, 0.5, 0.75, 0.9
            - "tukey": min, max, 0.125, 0.25, 0.5, 0.75, 0.875
    estimator : sklearn classifier, default=None
        An sklearn estimator to be built using the transformed data. Defaults to a
        Random Forest with 200 trees.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
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
    classes_ : ndarray of shape (n_classes)
        Holds the label for each class.
    estimator_ : sklearn classifier
        The fitted estimator.
    transformer_ : SevenNumberSummary
        The fitted transformer.

    See Also
    --------
    SummaryTransformer
    SummaryRegressor

    Examples
    --------
    >>> from aeon.classification.feature_based import SummaryClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = SummaryClassifier(estimator=RandomForestClassifier(n_estimators=5))
    >>> clf.fit(X_train, y_train)
    SummaryClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "feature",
    }

    def __init__(
        self,
        summary_stats="default",
        estimator=None,
        n_jobs=1,
        random_state=None,
        class_weight=None,
    ):
        self.summary_stats = summary_stats
        self.estimator = estimator

        self.n_jobs = n_jobs
        self.random_state = random_state

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
        self.transformer_ = SevenNumberSummary(
            summary_stats=self.summary_stats,
        )

        self.estimator_ = _clone_estimator(
            (
                RandomForestClassifier(n_estimators=200, class_weight=self.class_weight)
                if self.estimator is None
                else self.estimator
            ),
            self.random_state,
        )

        m = getattr(self.estimator_, "n_jobs", None)
        if m is not None:
            self.estimator_.n_jobs = self._n_jobs

        X_t = self.transformer_.fit_transform(X, y)
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
        return self.estimator_.predict(self.transformer_.transform(X))

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
        m = getattr(self.estimator_, "predict_proba", None)
        if callable(m):
            return self.estimator_.predict_proba(self.transformer_.transform(X))
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self.estimator_.predict(self.transformer_.transform(X))
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
            SummaryClassifier provides the following special sets:
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
            return {"estimator": RandomForestClassifier(n_estimators=10)}
        else:
            return {"estimator": RandomForestClassifier(n_estimators=2)}
