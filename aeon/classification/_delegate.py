# -*- coding: utf-8 -*-
"""Delegator mixin that delegates all methods to wrapped classifiers.

Useful for building estimators where all but one or a few methods are delegated.
For that purpose, inherit from this estimator and then override only the methods
    that are not delegated.
"""
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["_DelegatedClassifier"]

from aeon.classification.base import BaseClassifier


class _DelegatedClassifier(BaseClassifier):
    """
    Delegator mixin that delegates all methods to wrapped classifier.

    Delegates inner classifier methods to a wrapped estimator.
        Wrapped estimator is value of attribute with name self._delegate_name.
        By default, this is "estimator_", i.e., delegates to self.estimator_
        To override delegation, override _delegate_name attribute in child class.

    Delegates the following inner underscore methods:
        _fit, _predict, _predict_proba

    Does NOT delegate get_params, set_params.
        get_params, set_params will hence use one additional nesting level by default.

    Does NOT delegate or copy tags, this should be done in a child class if required.
    """

    # attribute for _Delegatedclassifier, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedClassifier docstring
    _delegate_name = "estimator_"

    def _get_delegate(self):
        return getattr(self, self._delegate_name)

    def _fit(self, X, y):
        """Fit time series classifier to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : 3D np.array
            Input data, any number of channels, equal length series of shape ``(
            n_instances, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_instances, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_instances]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. Other types are
            allowed and converted into one of the above.
        y: np.array
            shape ``(n_instances)`` - class labels for fitting indices correspond to
            instance indices in X.

        Returns
        -------
        self : Reference to self.
        """
        estimator = self._get_delegate()
        estimator.fit(X=X, y=y)
        return self

    def _predict(self, X):
        """Predict labels for sequences in X.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : 3D np.array
            Input data, any number of channels, equal length series of shape ``(
            n_instances, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_instances, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_instances]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. Other types are
            allowed and converted into one of the above.

        Returns
        -------
        y : 1D np.array of int, of shape [n_instances] - predicted class labels
            indices correspond to instance indices in X
        """
        estimator = self._get_delegate()
        return estimator.predict(X=X)

    def _predict_proba(self, X):
        """Predicts labels probabilities for sequences in X.

        private _predict_proba containing the core logic, called from predict_proba

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : 3D np.array
            Input data, any number of channels, equal length series of shape ``(
            n_instances, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_instances, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_instances]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. Other types are
            allowed and converted into one of the above.

        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        estimator = self._get_delegate()
        return estimator.predict_proba(X=X)

    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
        """
        estimator = self._get_delegate()
        return estimator.get_fitted_params()
