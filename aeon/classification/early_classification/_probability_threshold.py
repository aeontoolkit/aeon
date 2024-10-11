"""Probability Threshold Early Classifier.

An early classifier using a prediction probability threshold with a time series
classifier.
"""

__maintainer__ = []
__all__ = ["ProbabilityThresholdEarlyClassifier"]

import copy

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state

from aeon.base._base import _clone_estimator
from aeon.classification.early_classification.base import BaseEarlyClassifier
from aeon.classification.interval_based import DrCIFClassifier


class ProbabilityThresholdEarlyClassifier(BaseEarlyClassifier):
    """
    Probability Threshold Early Classifier.

    An early classifier which uses a threshold of prediction probability to determine
    whether an early prediction is safe or not.

    Overview:
        Build n classifiers, where n is the number of classification_points.
        While a prediction is still deemed unsafe:
            Make a prediction using the series length at classification point i.
            Decide whether the predcition is safe or not using decide_prediction_safety.

    Parameters
    ----------
    probability_threshold : float, default=0.85
        The class prediction probability required to deem a prediction as safe.
    consecutive_predictions : int, default=1
        The number of consecutive predictions for a class above the threshold required
        to deem a prediction as safe.
    estimator : aeon classifier, default=None
        An aeon estimator to be built using the transformed data. Defaults to a
        default DrCIF classifier.
    classification_points : List or None, default=None
        List of integer time series time stamps to build classifiers and allow
        predictions at. Early predictions must have a series length that matches a value
        in the _classification_points List. Duplicate values will be removed, and the
        full series length will be appeneded if not present.
        If None, will use 20 thresholds linearly spaces from 0 to the series length.
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
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    n_cases_ : int
        The number of train cases.
    n_channels_ : int
        The number of dimensions per case.
    n_timepoints_ : int
        The full length of each series.
    classes_ : list
        The unique class labels.
    state_info : 2d np.ndarray (4 columns)
        Information stored about input instances after the decision-making process in
        update/predict methods. Used in update methods to make decisions based on
        the resutls of previous method calls.
        Records in order: the time stamp index, the number of consecutive decisions
        made, the predicted class and the series length.

    Examples
    --------
    >>> from aeon.classification.early_classification import (
    ...     ProbabilityThresholdEarlyClassifier
    ... )
    >>> from aeon.classification.interval_based import TimeSeriesForestClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = ProbabilityThresholdEarlyClassifier(
    ...     classification_points=[6, 16, 24],
    ...     estimator=TimeSeriesForestClassifier(n_estimators=5),
    ... )
    >>> clf.fit(X_train, y_train)
    ProbabilityThresholdEarlyClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        estimator=None,
        probability_threshold=0.85,
        consecutive_predictions=1,
        classification_points=None,
        class_weight=None,
        n_jobs=1,
        random_state=None,
    ):
        self.estimator = estimator
        self.probability_threshold = probability_threshold
        self.consecutive_predictions = consecutive_predictions
        self.classification_points = classification_points

        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._estimators = []
        self._classification_points = []

        self.n_cases_ = 0
        self.n_channels_ = 0
        self.n_timepoints_ = 0

        super().__init__()

    def _fit(self, X, y):
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape

        self._estimator = (
            DrCIFClassifier() if self.estimator is None else self.estimator
        )

        m = getattr(self._estimator, "predict_proba", None)
        if not callable(m):
            raise ValueError("Base estimator must have a predict_proba method.")

        self._classification_points = (
            copy.deepcopy(self.classification_points)
            if self.classification_points is not None
            else [round(self.n_timepoints_ / i) for i in range(1, 21)]
        )
        # remove duplicates
        self._classification_points = list(set(self._classification_points))
        self._classification_points.sort()
        # remove classification points that are less than 3 time stamps
        self._classification_points = [i for i in self._classification_points if i >= 3]
        # make sure the full series length is included
        if self._classification_points[-1] != self.n_timepoints_:
            self._classification_points.append(self.n_timepoints_)
        # create dictionary of classification point indices
        self._classification_point_dictionary = {}
        for index, classification_point in enumerate(self._classification_points):
            self._classification_point_dictionary[classification_point] = index

        # avoid nested parallelism
        m = getattr(self._estimator, "n_jobs", None)
        threads = self._n_jobs if m is None else 1

        rng = check_random_state(self.random_state)

        self._estimators = Parallel(n_jobs=threads, prefer="threads")(
            delayed(self._fit_estimator)(
                X,
                y,
                i,
                check_random_state(rng.randint(np.iinfo(np.int32).max)),
            )
            for i in range(len(self._classification_points))
        )

        return self

    def _predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        out = self._predict_proba(X)
        return self._proba_output_to_preds(out)

    def _update_predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        out = self._update_predict_proba(X)
        return self._proba_output_to_preds(out)

    def _predict_proba(self, X) -> tuple[np.ndarray, np.ndarray]:
        n_cases, _, n_timepoints = X.shape

        # maybe use the largest index that is smaller than the series length
        next_idx = self._get_next_idx(n_timepoints) + 1

        # if the input series length is invalid
        if next_idx == 0:
            raise ValueError(
                f"Input series length does not match the classification points produced"
                f" in fit. Input series length must be greater then the first point. "
                f"Current classification points: {self._classification_points}"
            )

        # avoid nested parallelism
        m = getattr(self._estimator, "n_jobs", None)
        threads = self._n_jobs if m is None else 1

        rng = check_random_state(self.random_state)

        # compute all new updates since then
        out = Parallel(n_jobs=threads, prefer="threads")(
            delayed(self._predict_proba_for_estimator)(
                X,
                i,
                check_random_state(rng.randint(np.iinfo(np.int32).max)),
            )
            for i in range(0, next_idx)
        )
        probas, preds = zip(*out)

        # a List containing the state info for case, edited at each time stamp.
        # contains 1. the index of the time stamp, 2. the number of consecutive
        # positive decisions made, and 3. the prediction made
        self.state_info = np.zeros((len(preds[0]), 4), dtype=int)

        probas, accept_decision, self.state_info = self._decide_and_return_probas(
            0, next_idx, probas, preds, self.state_info
        )

        return probas, accept_decision

    def _update_predict_proba(self, X) -> tuple[np.ndarray, np.ndarray]:
        n_timepoints = X.shape[2]

        # maybe use the largest index that is smaller than the series length
        next_idx = self._get_next_idx(n_timepoints) + 1

        # remove cases where a positive decision has been made
        state_info = self.state_info[
            self.state_info[:, 1] < self.consecutive_predictions
        ]

        # determine last index used
        last_idx = np.max(state_info[0][0]) + 1

        # if the input series length is invalid
        if next_idx == 0:
            raise ValueError(
                f"Input series length does not match the classification points produced"
                f" in fit. Input series length must be greater then the first point. "
                f"Current classification points: {self._classification_points}"
            )
        # check state info and X have the same length
        if len(X) > len(state_info):
            raise ValueError(
                f"Input number of instances does not match the length of recorded "
                f"state_info: {len(state_info)}. Cases with positive decisions "
                f"returned should be removed from the array with the row ordering "
                f"preserved, or the state information should be reset if new data is "
                f"used."
            )
        # check if series length has increased from last time
        elif last_idx >= next_idx:
            raise ValueError(
                f"All input instances must be from a larger classification point time "
                f"stamp than the recorded state information. Required series length "
                f"for current state information: "
                f">={self._classification_points[last_idx]}"
            )

        # avoid nested parallelism
        m = getattr(self._estimator, "n_jobs", None)
        threads = self._n_jobs if m is None else 1

        rng = check_random_state(self.random_state)

        # compute all new updates since then
        out = Parallel(n_jobs=threads, prefer="threads")(
            delayed(self._predict_proba_for_estimator)(
                X,
                i,
                check_random_state(rng.randint(np.iinfo(np.int32).max)),
            )
            for i in range(last_idx, next_idx)
        )
        probas, preds = zip(*out)

        probas, accept_decision, self.state_info = self._decide_and_return_probas(
            last_idx, next_idx, probas, preds, state_info
        )

        return probas, accept_decision

    def _decide_and_return_probas(self, last_idx, next_idx, probas, preds, state_info):
        # only compute new indices
        for i in range(last_idx, next_idx):
            accept_decision, state_info = self._decide_prediction_safety(
                i,
                probas[i - last_idx],
                preds[i - last_idx],
                state_info,
            )

        probas = np.array(
            [
                (
                    probas[max(0, state_info[i][0] - last_idx)][i]
                    if accept_decision[i]
                    else [-1 for _ in range(self.n_classes_)]
                )
                for i in range(len(accept_decision))
            ]
        )

        return probas, accept_decision, state_info

    def _score(self, X, y) -> tuple[float, float, float]:
        self._predict(X)
        hm, acc, earl = self.compute_harmonic_mean(self.state_info, y)

        return hm, acc, earl

    def _decide_prediction_safety(self, idx, probas, preds, state_info):
        # stores whether we have made a final decision on a prediction, if true
        # state info won't be edited in later time stamps
        finished = state_info[:, 1] >= self.consecutive_predictions
        n_cases = len(preds)

        full_length_ts = idx == len(self._classification_points) - 1
        if full_length_ts:
            accept_decision = np.ones(n_cases, dtype=bool)
        else:
            offsets = np.argwhere(finished == 0).flatten()
            accept_decision = np.ones(n_cases, dtype=bool)
            if len(offsets) > 0:
                p = probas[offsets, preds[offsets]]
                accept_decision[offsets] = p >= self.probability_threshold

        # record consecutive class decisions
        state_info = np.array(
            [
                (
                    self._update_state_info(accept_decision, preds, state_info, i, idx)
                    if not finished[i]
                    else state_info[i]
                )
                for i in range(n_cases)
            ]
        )

        # check safety of decisions
        if full_length_ts:
            # Force prediction at last time stamp
            accept_decision = np.ones(n_cases, dtype=bool)
        else:
            accept_decision = state_info[:, 1] >= self.consecutive_predictions

        return accept_decision, state_info

    def _fit_estimator(self, X, y, i, rng):
        estimator = _clone_estimator(
            self._estimator,
            rng,
        )

        m = getattr(estimator, "n_jobs", None)
        if m is not None:
            estimator.n_jobs = self._n_jobs

        estimator.fit(X[:, :, : self._classification_points[i]], y)

        return estimator

    def _predict_proba_for_estimator(self, X, i, rng):
        probas = self._estimators[i].predict_proba(
            X[:, :, : self._classification_points[i]]
        )
        preds = np.array(
            [int(rng.choice(np.flatnonzero(prob == prob.max()))) for prob in probas]
        )

        return probas, preds

    def _get_next_idx(self, n_timepoints):
        """Return the largest index smaller than the series length."""
        next_idx = -1
        for idx, offset in enumerate(np.sort(self._classification_points)):
            if offset <= n_timepoints:
                next_idx = idx
        return next_idx

    def _update_state_info(self, accept_decision, preds, state_info, idx, time_stamp):
        # consecutive predictions, add one if positive decision and same class
        if accept_decision[idx] and preds[idx] == state_info[idx][2]:
            return (
                time_stamp,
                state_info[idx][1] + 1,
                preds[idx],
                self._classification_points[time_stamp],
            )
        # set to 0 if the decision is negative, 1 if its positive but different class
        else:
            return (
                time_stamp,
                1 if accept_decision[idx] else 0,
                preds[idx],
                self._classification_points[time_stamp],
            )

    def _proba_output_to_preds(self, out):
        rng = check_random_state(self.random_state)
        preds = np.array(
            [
                (
                    self.classes_[
                        int(rng.choice(np.flatnonzero(out[0][i] == out[0][i].max())))
                    ]
                    if out[1][i]
                    else -1
                )
                for i in range(len(out[0]))
            ]
        )
        return preds, out[1]

    def compute_harmonic_mean(self, state_info, y) -> tuple[float, float, float]:
        """Calculate harmonic mean from a state info matrix and array of class labeles.

        Parameters
        ----------
        state_info : 2d np.ndarray of int
            The state_info from a ProbabilityThresholdEarlyClassifier object after a
            prediction or update. It is assumed the state_info is complete, and a
            positive decision has been returned for all cases.
        y : 1D np.array of int
            Actual class labels for predictions. indices correspond to instance indices
            in state_info.


        Returns
        -------
        harmonic_mean : float
            Harmonic Mean represents the balance between accuracy and earliness for a
            set of early predictions.
        accuracy : float
            Accuracy for the predictions made in the state_info.
        earliness : float
            Average time taken to make a classification. The earliness for a single case
            is the number of time points required divided by the total series length.
        """
        accuracy = np.average(
            [
                state_info[i][2] == self._class_dictionary[y[i]]
                for i in range(len(state_info))
            ]
        )
        earliness = np.average(
            [
                self._classification_points[state_info[i][0]] / self.n_timepoints_
                for i in range(len(state_info))
            ]
        )
        return (
            (2 * accuracy * (1 - earliness)) / (accuracy + (1 - earliness)),
            accuracy,
            earliness,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            ProbabilityThresholdEarlyClassifier provides the following special sets:
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
        from aeon.classification.feature_based import SummaryClassifier
        from aeon.classification.interval_based import TimeSeriesForestClassifier

        if parameter_set == "results_comparison":
            return {
                "classification_points": [6, 10, 16, 24],
                "estimator": TimeSeriesForestClassifier(n_estimators=10),
            }
        else:
            return {
                "classification_points": [3, 5],
                "estimator": SummaryClassifier(
                    estimator=RandomForestClassifier(n_estimators=2)
                ),
            }
