"""aeon discovery methods."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["all_estimators"]

import inspect
import warnings
from importlib import import_module
from operator import itemgetter
from pathlib import Path
from pkgutil import walk_packages

from sklearn.base import BaseEstimator

from aeon.base import BaseAeonEstimator
from aeon.utils.base import BASE_CLASS_REGISTER
from aeon.utils.tags import ESTIMATOR_TAGS
from aeon.utils.tags._validate import check_tag_value


def all_estimators(
    type_filter=None,
    exclude_types=None,
    tag_filter=None,
    exclude_tags=None,
    include_sklearn=True,
    return_names=True,
):
    """Get a list of all estimators from aeon.

    This function crawls the module and gets all classes that inherit
    from aeon's and sklearn's base classes.

    Not included are: the base classes themselves, classes defined in test
    modules.

    Based on the sklearn utility of the same name.

    Parameters
    ----------
    type_filter: str, estimator type or list of str/type, default=None
        Which kind of estimators should be returned.
        if None, no filter is applied.
        if str, type or list, returns esimators that are of one of the provided types.
            Valid entries are available in the keys and values of
            utils.base.BASE_CLASS_REGISTER. i.e. "classifier" and "transformer" for
            str, and BaseClassifier and BaseTransformer for type.
    exclude_types: str, estimator type or list of str/type, default=None
        Which kind of estimators to exclude. Inverse of type_filter.
        if None, no exclusion is applied.
        if str, type or list, excludes esimators that are of the provided types.
            Valid entries are available in the keys and values of
            utils.base.BASE_CLASS_REGISTER. i.e. "classifier" and "transformer" for
            str, and BaseClassifier and BaseTransformer for type.
    tag_filter: dict of (str or list of str), default=None
        Return only estimators that have tags matching the provided values.
            i.e. {"capability:unequal_length":True} will return all estimators that
            have the tag "capability:unequal_length" with a value of True.
    exclude_tags: dict of (str or list of str), default=None
        Exclude only estimators that have tags matching the provided values. Inverse
        of tag_filter.
            i.e. {"capability:unequal_length":True} will exclude all estimators that
            have the tag "capability:unequal_length" with a value of True.
    include_sklearn: bool, default=True
        if True, include estimators implemented using the sklearn interface (not
        inheriting from BaseAeonEstimator) in the output. type_filter and tag_filter
        will remove these estimators.
    return_names: bool, default=True
        if True, a tuple (estimator name, estimator class) is returned for each
            estimator in the list.
        if False, only the estimator class is returned.

    Returns
    -------
    estimators
        list of tuples (estimator name, estimator class) if return_names=True,
        else list of estimator classes

    Examples
    --------
    >>> from aeon.utils.discovery import all_estimators
    >>> # return a complete list of estimators as pd.Dataframe
    >>> all = all_estimators()
    >>> # return all classifiers by filtering for estimator type
    >>> classifiers = all_estimators(type_filter="classifier")
    >>> # return all classifiers which handle unequal length data by tag filtering
    >>> clf_ul = all_estimators(
    ...     type_filter="classifier", tag_filter={"capability:unequal_length":True}
    ... )
    """
    modules_to_ignore = (
        # ignore test modules and base classes
        "base",
        "tests",
        "similarity_search"
        # ignore these submodules
        "benchmarking",
        "datasets",
        "distances",
        "networks",
        "performance_metrics",
        "pipeline",
        "testing",
        "utils",
        "visualisation",
    )
    base_filter = BaseEstimator if include_sklearn else BaseAeonEstimator
    root = str(Path(__file__).parent.parent)  # aeon package root directory
    estimators = []

    # ignore deprecation warnings triggered at import time and from walking packages
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        for _, module_name, _ in walk_packages(path=[root], prefix="aeon."):
            module_parts = module_name.split(".")
            if (
                any(part in modules_to_ignore for part in module_parts)
                or "._" in module_name
            ):
                continue
            module = import_module(module_name)

            classes = inspect.getmembers(module, inspect.isclass)
            # skip private estimators and those not implemented in aeon
            classes = [
                (name, est_cls)
                for name, est_cls in classes
                if not name.startswith("_")
                and str(est_cls).startswith("<class 'aeon")
                and issubclass(est_cls, base_filter)
            ]

            estimators.extend(classes)

    # drop duplicates
    estimators = set(estimators)
    # drop abstract classes
    estimators = [c for c in estimators if not _is_abstract(c[1])]

    # filter based on wanted/excluded
    if type_filter is not None:
        estimators = _filter_types(type_filter, estimators, "type_filter")
    if exclude_types is not None:
        estimators = set(estimators) - set(
            _filter_types(exclude_types, estimators, "exclude_types")
        )

    # filter based on wanted/excluded tags
    if tag_filter is not None:
        estimators = _filter_tags(tag_filter, estimators, "tag_filter")
    if exclude_tags is not None:
        estimators = set(estimators) - set(
            _filter_tags(exclude_tags, estimators, "exclude_tags")
        )

    # sort for reproducibility, remove names if return_names=False
    estimators = sorted(set(estimators), key=itemgetter(0))
    if not return_names:
        return [est for (name, est) in estimators]
    else:
        return estimators


def _is_abstract(c):
    if not (hasattr(c, "__abstractmethods__")):
        return False
    if not len(c.__abstractmethods__):
        return False
    return True


def _filter_types(types, estimators, name):
    msg = (
        f"Parameter {name} must be None, a string or type, or a list of "
        f"strings or types. Valid string/type values are: "
        f"{BASE_CLASS_REGISTER}. Found: {types}"
    )

    if not isinstance(types, list):
        types = [types]
    else:
        types = list(types)  # copy

    filtered_estimators = []
    for t in types:
        if isinstance(t, str):
            if t not in BASE_CLASS_REGISTER.keys():
                raise ValueError(msg)
            filtered_estimators.extend(
                [
                    est
                    for est in estimators
                    if issubclass(est[1], BASE_CLASS_REGISTER[t])
                ]
            )
        elif isinstance(t, type):
            if t not in BASE_CLASS_REGISTER.values():
                raise ValueError(msg)
            filtered_estimators.extend(
                [est for est in estimators if issubclass(est[1], t)]
            )
        else:
            raise ValueError(msg)

    return filtered_estimators


def _filter_tags(tags, estimators, name):
    if tags is None:
        return estimators

    msg = (
        f"Parameter {name} must be None or a dict of tag/value pairs. "
        f"Valid tags are found in aeon.utils.tags.ESTIMATOR_TAGS. Found: {tags}"
    )

    if not isinstance(tags, dict):
        raise TypeError(msg)

    for key, value in tags.items():
        if key not in ESTIMATOR_TAGS:
            raise ValueError(f"Tag {key} is not a valid tag. {msg}")
        if not check_tag_value(key, value, raise_error=False):
            raise ValueError(f"Value {value} is not a valid value for tag {key}. {msg}")

    filtered_estimators = []
    for est in estimators:
        if not issubclass(est[1], BaseAeonEstimator):
            continue

        cond_sat = True
        for key, value in tags.items():
            est_tag = est[1].get_class_tag(key, raise_error=False)
            est_tag = est_tag if isinstance(est_tag, list) else [est_tag]

            if isinstance(value, list):
                in_list = False
                for n in value:
                    if n in est_tag:
                        in_list = True
                cond_sat = cond_sat and in_list
            else:
                cond_sat = cond_sat and value in est_tag

        if cond_sat:
            filtered_estimators.append(est)

    return filtered_estimators
