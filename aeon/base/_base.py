"""Base class template for aeon estimators."""

__maintainer__ = ["MatthewMiddlehurst", "TonyBagnall"]
__all__ = ["BaseAeonEstimator"]

import inspect
from abc import ABC, abstractmethod
from copy import deepcopy

from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.ensemble._base import _set_random_states
from sklearn.exceptions import NotFittedError

from aeon.utils.validation._dependencies import _check_estimator_deps


class BaseAeonEstimator(BaseEstimator, ABC):
    """
    Base class for defining estimators in aeon.

    Contains the following methods:

    - reset estimator to post-init  - reset(keep)
    - clone stimator (copy)         - clone(random_state)
    - inspect tags (class method)   - get_class_tags()
    - inspect tags (one tag, class) - get_class_tag(tag_name, tag_value_default,
                                                                    raise_error)
    - inspect tags (all)            - get_tags()
    - inspect tags (one tag)        - get_tag(tag_name, tag_value_default, raise_error)
    - setting dynamic tags          - set_tags(**tag_dict)
    - get fitted parameters         - get_fitted_params(deep)

    All estimators have the attribute:

    - fitted state flag             - is_fitted
    """

    _tags = {
        "python_version": None,
        "python_dependencies": None,
        "cant_pickle": False,
        "non_deterministic": False,
        "algorithm_type": None,
        "capability:missing_values": False,
        "capability:multithreading": False,
    }

    @abstractmethod
    def __init__(self):
        self.is_fitted = False  # flag to indicate if fit has been called
        self._tags_dynamic = dict()  # storage for dynamic tags

        super().__init__()

        _check_estimator_deps(self)

    def reset(self, keep=None):
        """
        Reset the object to a clean post-init state.

        After a ``self.reset()`` call, self is equal or similar in value to
        ``type(self)(**self.get_params(deep=False))``, assuming no other attributes
        were kept using ``keep``.

        Detailed behaviour:
            removes any object attributes, except:
                hyper-parameters (arguments of ``__init__``)
                object attributes containing double-underscores, i.e., the string "__"
            runs ``__init__`` with current values of hyperparameters (result of
            ``get_params``)

        Not affected by the reset are:
            object attributes containing double-underscores
            class and object methods, class attributes
            any attributes specified in the ``keep`` argument

        Parameters
        ----------
        keep : None, str, or list of str, default=None
            If None, all attributes are removed except hyperparameters.
            If str, only the attribute with this name is kept.
            If list of str, only the attributes with these names are kept.

        Returns
        -------
        self : object
            Reference to self.
        """
        # retrieve parameters to copy them later
        params = self.get_params(deep=False)

        # delete all object attributes in self
        attrs = [attr for attr in dir(self) if "__" not in attr]
        cls_attrs = [attr for attr in dir(type(self))]
        self_attrs = set(attrs).difference(cls_attrs)

        # keep specific attributes if set
        if keep is not None:
            if isinstance(keep, str):
                keep = [keep]
            elif not isinstance(keep, list):
                raise TypeError(
                    "keep must be a string or list of strings containing attributes "
                    "to keep after the reset."
                )
            for attr in keep:
                self_attrs.discard(attr)

        for attr in self_attrs:
            delattr(self, attr)

        # run init with a copy of parameters self had at the start
        self.__init__(**params)

        return self

    def clone(self, random_state=None):
        """
        Obtain a clone of the object with the same hyperparameters.

        A clone is a different object without shared references, in post-init state.
        This function is equivalent to returning ``sklearn.clone`` of self.
        Equal in value to ``type(self)(**self.get_params(deep=False))``.

        Parameters
        ----------
        random_state : int, RandomState instance, or None, default=None
            Sets the random state of the clone. If None, the random state is not set.
            If int, random_state is the seed used by the random number generator.
            If RandomState instance, random_state is the random number generator.

        Returns
        -------
        estimator : object
            Instance of ``type(self)``, clone of self (see above)
        """
        estimator = clone(self)

        if random_state is not None:
            _set_random_states(estimator, random_state)

        return estimator

    @classmethod
    def get_class_tags(cls):
        """
        Get class tags from estimator class and all its parent classes.

        Returns
        -------
        collected_tags : dict
            Dictionary of tag name and tag value pairs.
            Collected from ``_tags`` class attribute via nested inheritance.
            These are not overridden by dynamic tags set by ``set_tags`` or class
            ``__init__`` calls.
        """
        collected_tags = dict()

        # We exclude the last two parent classes: sklearn.base.BaseEstimator and
        # the basic Python object.
        for parent_class in reversed(inspect.getmro(cls)[:-2]):
            # Need the if here because classes might not have non-default tags
            if hasattr(parent_class, "_tags"):
                more_tags = parent_class._tags
                collected_tags.update(more_tags)

        return deepcopy(collected_tags)

    @classmethod
    def get_class_tag(
        cls,
        tag_name,
        raise_error=True,
        tag_value_default=None,
    ):
        """
        Get tag value from estimator class (only class tags).

        Parameters
        ----------
        tag_name : str
            Name of tag value.
        raise_error : bool, default=True
            Whether a ValueError is raised when the tag is not found.
        tag_value_default : any type, default=None
            Default/fallback value if tag is not found and error is not raised.

        Returns
        -------
        tag_value
            Value of the ``tag_name`` tag in cls.
            If not found, returns an error if ``raise_error`` is True, otherwise it
            returns ``tag_value_default``.

        Raises
        ------
        ValueError
            if ``raise_error`` is True and ``tag_name`` is not in
            ``self.get_tags().keys()``

        Examples
        --------
        >>> from aeon.classification import DummyClassifier
        >>> DummyClassifier.get_class_tag("capability:multivariate")
        True
        """
        collected_tags = cls.get_class_tags()

        tag_value = collected_tags.get(tag_name, tag_value_default)

        if raise_error and tag_name not in collected_tags.keys():
            raise ValueError(f"Tag with name {tag_name} could not be found.")

        return tag_value

    def get_tags(self):
        """
        Get tags from estimator.

        Includes dynamic and overridden tags.

        Returns
        -------
        collected_tags : dict
            Dictionary of tag name and tag value pairs.
            Collected from ``_tags`` class attribute via nested inheritance and
            then any overridden and new tags from ``__init__`` or ``set_tags``.
        """
        collected_tags = self.get_class_tags()
        collected_tags.update(self._tags_dynamic)
        return deepcopy(collected_tags)

    def get_tag(self, tag_name, raise_error=True, tag_value_default=None):
        """
        Get tag value from estimator class.

        Includes dynamic and overridden tags.

        Parameters
        ----------
        tag_name : str
            Name of tag to be retrieved.
        raise_error : bool, default=True
            Whether a ValueError is raised when the tag is not found.
        tag_value_default : any type, default=None
            Default/fallback value if tag is not found and error is not raised.

        Returns
        -------
        tag_value
            Value of the ``tag_name`` tag in self.
            If not found, returns an error if ``raise_error`` is True, otherwise it
            returns ``tag_value_default``.

        Raises
        ------
        ValueError
            if raise_error is ``True`` and ``tag_name`` is not in
            ``self.get_tags().keys()``

        Examples
        --------
        >>> from aeon.classification import DummyClassifier
        >>> d = DummyClassifier()
        >>> d.get_tag("capability:multivariate")
        True
        """
        collected_tags = self.get_tags()

        tag_value = collected_tags.get(tag_name, tag_value_default)

        if raise_error and tag_name not in collected_tags.keys():
            raise ValueError(f"Tag with name {tag_name} could not be found.")

        return tag_value

    def set_tags(self, **tag_dict):
        """
        Set dynamic tags to given values.

        Parameters
        ----------
        **tag_dict : dict
            Dictionary of tag name and tag value pairs.

        Returns
        -------
        self : object
            Reference to self.
        """
        tag_update = deepcopy(tag_dict)
        self._tags_dynamic.update(tag_update)
        return self

    def get_fitted_params(self, deep=True):
        """Get fitted parameters.

        State required:
            Requires state to be "fitted".

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the fitted parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        fitted_params : dict
            Fitted parameter names mapped to their values.
        """
        self._check_is_fitted()
        return self._get_fitted_params(self, deep)

    def _get_fitted_params(self, est, deep):
        """Recursive function to get fitted parameters."""
        # retrieves all self attributes ending in "_"
        fitted_params = [
            attr for attr in dir(est) if attr.endswith("_") and not attr.startswith("_")
        ]

        out = dict()
        for key in fitted_params:
            # some of these can be properties and can make assumptions which may not be
            # true in aeon i.e. sklearn Pipeline feature_names_in_
            try:
                value = getattr(est, key)
            except AttributeError:
                continue

            if deep and isinstance(value, BaseEstimator):
                deep_items = self._get_fitted_params(value, deep).items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    # private functions to help testing

    def _check_is_fitted(self):
        """
        Check if the estimator has been fitted.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class. Each dict are
            parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        # default parameters = empty dict
        return {}

    @classmethod
    def _create_test_instance(cls, parameter_set="default", return_first=True):
        """
        Construct Estimator instance if possible.

        Calls the `_get_test_params` method and returns an instance or list of instances
        using the returned dict or list of dict.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
        return_first : bool, default=True
            If True, return the first instance of the list of instances.
            If False, return the list of instances.

        Returns
        -------
        instance : BaseAeonEstimator or list of BaseAeonEstimator
            Instance of the class with default parameters. If return_first
            is False, returns list of instances.
        """
        params = cls._get_test_params(parameter_set=parameter_set)

        if isinstance(params, list):
            if return_first:
                return cls(**params[0])
            else:
                return [cls(**p) for p in params]
        else:
            if return_first:
                return cls(**params)
            else:
                return [cls(**params)]

    # override some sklearn private methods

    def __sklearn_is_fitted__(self):
        """Check fitted status and return a Boolean value."""
        return self.is_fitted

    def __sklearn_tags__(self):
        """Return sklearn style tags for the estimator."""
        aeon_tags = self.get_tags()
        sklearn_tags = super().__sklearn_tags__()
        sklearn_tags.non_deterministic = aeon_tags.get("non_deterministic", False)
        sklearn_tags.target_tags.one_d_labels = True
        sklearn_tags.input_tags.three_d_array = True
        sklearn_tags.input_tags.allow_nan = aeon_tags.get(
            "capability:missing_values", False
        )
        return sklearn_tags

    def _validate_data(self, **kwargs):
        """Sklearn data validation."""
        raise NotImplementedError(
            "aeon estimators do not have a _validate_data method."
        )

    def get_metadata_routing(self):
        """Sklearn metadata routing.

        Not supported by ``aeon`` estimators.
        """
        raise NotImplementedError(
            "aeon estimators do not have a get_metadata_routing method."
        )

    @classmethod
    def _get_default_requests(cls):
        """Sklearn metadata request defaults."""
        from sklearn.utils._metadata_requests import MetadataRequest

        return MetadataRequest(None)


def _clone_estimator(base_estimator, random_state=None):
    """Clone an estimator."""
    estimator = clone(base_estimator)

    if random_state is not None:
        _set_random_states(estimator, random_state)

    return estimator
