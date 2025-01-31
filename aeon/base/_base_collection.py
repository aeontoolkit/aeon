"""
Base class for estimators that fit collections of time series.

    class name: BaseCollectionEstimator

Defining methods:
    preprocessing         - _preprocess_collection(self, X, store_metadata=True)
    input checking        - _check_X(self, X)
    input conversion      - _convert_X(self, X)
    shape checking        - _check_shape(self, X)

Inherited inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()

"""

from abc import abstractmethod

import numpy as np

from aeon.base._base import BaseAeonEstimator
from aeon.utils.conversion import (
    convert_collection,
    resolve_equal_length_inner_type,
    resolve_unequal_length_inner_type,
)
from aeon.utils.validation import check_n_jobs
from aeon.utils.validation.collection import (
    get_n_cases,
    get_n_channels,
    get_n_timepoints,
    get_type,
    has_missing,
    is_equal_length,
    is_univariate,
)


class BaseCollectionEstimator(BaseAeonEstimator):
    """
    Base class for estimators that use collections of time series for ``fit``.

    Provides functions that are common to estimators which use collections such as
    ``BaseClassifier``, ``BaseRegressor``, ``BaseClusterer``, ``BaseSimilaritySearch``
    and ``BaseCollectionTransformer``. Functionality includes checking and
    conversion of input in ``fit``, ``predict`` and ``predict_proba``, where relevant.

    It also stores the common default tags used by all the subclasses and meta data
    describing the characteristics of time series passed to ``fit``.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "X_inner_type": "numpy3D",
    }

    @abstractmethod
    def __init__(self):
        self.metadata_ = {}  # metadata/properties of data seen in fit
        self._n_jobs = 1

        super().__init__()

    def _preprocess_collection(self, X, store_metadata=True):
        """
        Preprocess input X prior to calling fit.

        1. Checks the characteristics of X and that self can handle the data
        2. Stores metadata about X in self.metadata_ if store_metadata is True
        3. Converts X to X_inner_type if necessary

        Parameters
        ----------
        X : collection
            See aeon.utils.COLLECTIONS_DATA_TYPES for details on aeon supported
            data structures.
        store_metadata : bool, default=True
            Whether to store metadata about X in self.metadata_.

        Returns
        -------
        X : collection
            Processed X. A data structure of type self.get_tag("X_inner_type").

        Raises
        ------
        ValueError
            If X is an invalid type or has characteristics that the estimator cannot
            handle.

        See Also
        --------
        _check_X :
            Function that checks X is valid before conversion.
        _convert_X :
            Function that converts to inner type.

        Examples
        --------
        >>> from aeon.testing.mock_estimators import MockClassifier
        >>> from aeon.testing.data_generation import make_example_2d_numpy_collection
        >>> clf = MockClassifier()
        >>> X, _ = make_example_2d_numpy_collection(n_cases=10, n_timepoints=20)
        >>> X2 = clf._preprocess_collection(X)
        >>> X2.shape
        (10, 1, 20)
        """
        if isinstance(X, list) and isinstance(X[0], np.ndarray):
            X = self._reshape_np_list(X)
        meta = self._check_X(X)
        if len(self.metadata_) == 0 and store_metadata:
            self.metadata_ = meta

        X = self._convert_X(X)
        # This usage of n_jobs is legacy, see issue #102
        multithread = self.get_tag("capability:multithreading")
        if multithread:
            if hasattr(self, "n_jobs"):
                self._n_jobs = check_n_jobs(self.n_jobs)
            else:
                raise AttributeError(
                    "self.n_jobs must be set if capability:multithreading is True"
                )
        return X

    def _check_X(self, X):
        """
        Check classifier input X is valid.

        Check if the input data is a compatible type, and that this estimator is
        able to handle the data characteristics.
        This is done by matching the capabilities of the estimator against the metadata
        for X i.e., univariate/multivariate, equal length/unequal length and no missing
        values/missing values.

        Parameters
        ----------
        X : collection
           See aeon.utils.COLLECTIONS_DATA_TYPES for details on aeon supported
           data structures.

        Returns
        -------
        metadata : dict
            Metadata about X, with flags:
            metadata["multivariate"] : whether X has more than one channel or not
            metadata["missing_values"] : whether X has missing values or not
            metadata["unequal_length"] : whether X contains unequal length series.
            metadata["n_cases"] : number of cases in X
            metadata["n_channels"] : number of channels in X
            metadata["n_timepoints"] : number of timepoints in X if equal length, else
                None

        Raises
        ------
        ValueError
            If X is an invalid type or has characteristics that the estimator cannot
            handle.

        See Also
        --------
        _convert_X :
            Function that converts X after it has been checked.

        Examples
        --------
        >>> from aeon.testing.mock_estimators import MockClassifierFullTags
        >>> from aeon.testing.data_generation import make_example_3d_numpy
        >>> clf = MockClassifierFullTags()
        >>> X, _ = make_example_3d_numpy(n_channels=3) # X is equal length, multivariate
        >>> meta = clf._check_X(X) # Classifier can handle this
        """
        # check if X is a valid type
        get_type(X)

        metadata = self._get_X_metadata(X)
        # Check classifier capabilities for X
        allow_multivariate = self.get_tag("capability:multivariate")
        allow_missing = self.get_tag("capability:missing_values")
        allow_unequal = self.get_tag("capability:unequal_length")

        # Check capabilities vs input
        problems = []
        if metadata["missing_values"] and not allow_missing:
            problems += ["missing values"]
        if metadata["multivariate"] and not allow_multivariate:
            problems += ["multivariate series"]
        if metadata["unequal_length"] and not allow_unequal:
            problems += ["unequal length series"]

        if problems:
            # construct error message
            problems_and = " and ".join(problems)
            msg = (
                f"Data seen by instance of {type(self).__name__} has {problems_and}, "
                f"but {type(self).__name__} cannot handle these characteristics. "
            )
            raise ValueError(msg)

        return metadata

    def _convert_X(self, X):
        """
        Convert X to type defined by tag X_inner_type.

        If the input data is already an allowed type, it is returned unchanged.

        If multiple types are allowed by self, then the best one for the type of input
        data is selected. So, for example, if X_inner_tag is ["np-list", "numpy3D"]
        and an df-list is passed, it will be converted to numpy3D if the series
        are equal length, and np-list if the series are unequal length.

        Parameters
        ----------
        X : collection
           See aeon.utils.COLLECTIONS_DATA_TYPES for details on aeon supported
           data structures.

        Returns
        -------
        X : collection
            Converted X. A data structure of type self.get_tag("X_inner_type").

        See Also
        --------
        _check_X :
            Function that checks X is valid and finds metadata.

        Examples
        --------
        >>> from aeon.testing.mock_estimators import MockClassifier
        >>> from aeon.testing.data_generation import make_example_3d_numpy_list
        >>> from aeon.utils.validation import get_type
        >>> clf = MockClassifier()
        >>> X, _ = make_example_3d_numpy_list(max_n_timepoints=8)
        >>> get_type(X)
        'np-list'
        >>> clf.get_tag("X_inner_type")
        'numpy3D'
        >>> X2 = clf._convert_X(X)
        >>> get_type(X2)
        'numpy3D'
        """
        inner_type = self.get_tag("X_inner_type")
        if not isinstance(inner_type, list):
            inner_type = [inner_type]
        input_type = get_type(X)

        # Check if we need to convert X, return if not
        if input_type in inner_type:
            return X

        if len(self.metadata_) == 0:
            metadata = self._get_X_metadata(X)
        else:
            metadata = self.metadata_

        # Convert X to X_inner_type if possible
        # If self can handle more than one internal type, resolve correct conversion
        # If unequal, choose data structure that can hold unequal
        if metadata["unequal_length"]:
            inner_type = resolve_unequal_length_inner_type(inner_type)
        else:
            inner_type = resolve_equal_length_inner_type(inner_type)

        return convert_collection(X, inner_type)

    def _check_shape(self, X):
        """
        Check that the shape of X is consistent with the data seen in fit.

        Parameters
        ----------
        X : data structure
            Must be of type aeon.registry.COLLECTIONS_DATA_TYPES.

        Raises
        ------
        ValueError
            If the shape of X is not consistent with the data seen in fit.
        """
        # if metadata is empty, then we have not seen any data in fit. If the estimator
        # has not been fitted, then _is_fitted should catch this.
        # there are valid cases where metadata is empty and the estimator has been
        # fitted, i.e. deep learner loading.
        if len(self.metadata_) != 0:
            if not self.get_tag("capability:unequal_length"):
                nt = get_n_timepoints(X)
                if nt != self.metadata_["n_timepoints"]:
                    raise ValueError(
                        "X has different length to the data seen in fit but "
                        "this classifier cannot handle unequal length series."
                        f"length of train set was {self.metadata_['n_timepoints']}",
                        f" length in predict is {nt}.",
                    )
            if self.get_tag("capability:multivariate"):
                nc = get_n_channels(X)
                if nc != self.metadata_["n_channels"]:
                    raise ValueError(
                        "X has different number of channels to the data seen in fit "
                        "number of channels in train set was ",
                        f"{self.metadata_['n_channels']} but in predict it is {nc}.",
                    )

    @staticmethod
    def _get_X_metadata(X):
        # Get and store X meta data.
        metadata = {}
        metadata["multivariate"] = not is_univariate(X)
        metadata["missing_values"] = has_missing(X)
        metadata["unequal_length"] = not is_equal_length(X)
        metadata["n_cases"] = get_n_cases(X)
        metadata["n_channels"] = get_n_channels(X)
        metadata["n_timepoints"] = (
            None if metadata["unequal_length"] else get_n_timepoints(X)
        )
        return metadata

    @staticmethod
    def _reshape_np_list(X):
        """Reshape 1D numpy to be 2D."""
        reshape = False
        for x in X:
            if x.ndim == 1:
                reshape = True
                break
        if reshape:
            X2 = []
            for x in X:
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                X2.append(x)
            return X2
        return X
