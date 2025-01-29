"""Tests for all estimators."""

import math
import numbers
import pickle
from copy import deepcopy
from functools import partial
from inspect import getfullargspec, isclass, signature

import joblib
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.exceptions import NotFittedError

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.base import BaseAeonEstimator
from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from aeon.classification.deep_learning.base import BaseDeepClassifier
from aeon.classification.early_classification import BaseEarlyClassifier
from aeon.clustering import BaseClusterer
from aeon.clustering.deep_learning.base import BaseDeepClusterer
from aeon.regression import BaseRegressor
from aeon.regression.deep_learning.base import BaseDeepRegressor
from aeon.segmentation import BaseSegmenter
from aeon.testing.estimator_checking._yield_anomaly_detection_checks import (
    _yield_anomaly_detection_checks,
)
from aeon.testing.estimator_checking._yield_classification_checks import (
    _yield_classification_checks,
)
from aeon.testing.estimator_checking._yield_clustering_checks import (
    _yield_clustering_checks,
)
from aeon.testing.estimator_checking._yield_early_classification_checks import (
    _yield_early_classification_checks,
)
from aeon.testing.estimator_checking._yield_multithreading_checks import (
    _yield_multithreading_checks,
)
from aeon.testing.estimator_checking._yield_regression_checks import (
    _yield_regression_checks,
)
from aeon.testing.estimator_checking._yield_segmentation_checks import (
    _yield_segmentation_checks,
)
from aeon.testing.estimator_checking._yield_soft_dependency_checks import (
    _yield_soft_dependency_checks,
)
from aeon.testing.estimator_checking._yield_transformation_checks import (
    _yield_transformation_checks,
)
from aeon.testing.testing_config import (
    NON_STATE_CHANGING_METHODS,
    NON_STATE_CHANGING_METHODS_ARRAYLIKE,
)
from aeon.testing.testing_data import FULL_TEST_DATA_DICT, _get_datatypes_for_estimator
from aeon.testing.utils.deep_equals import deep_equals
from aeon.testing.utils.estimator_checks import _get_tag, _run_estimator_method
from aeon.transformations.base import BaseTransformer
from aeon.utils.base import VALID_ESTIMATOR_BASES
from aeon.utils.tags import check_valid_tags
from aeon.utils.validation._dependencies import _check_estimator_deps


def _yield_all_aeon_checks(
    estimator, use_first_parameter_set=False, has_dependencies=None
):
    """Yield all checks for an aeon estimator."""
    # functions which use this will generally skip if dependencies are not met
    # UNLESS the check name has "softdep" in it
    if has_dependencies is None:
        has_dependencies = _check_estimator_deps(estimator, severity="none")

    if has_dependencies:
        if isclass(estimator) and issubclass(estimator, BaseAeonEstimator):
            estimator_class = estimator
            estimator_instances = estimator._create_test_instance(
                return_first=use_first_parameter_set
            )
        elif isinstance(estimator, BaseAeonEstimator):
            estimator_class = type(estimator)
            estimator_instances = estimator
        else:
            raise TypeError(
                "Passed estimator is not an instance or subclass of BaseAeonEstimator."
            )

        if not isinstance(estimator_instances, list):
            estimator_instances = [estimator_instances]

        datatypes = [_get_datatypes_for_estimator(est) for est in estimator_instances]
    else:
        # if input does not have all dependencies installed, all tests are going to be
        # skipped as we cannot instantiate the class
        # we still need inputs for the checks to return them and show that they
        # have been skipped
        estimator_class = estimator if isclass(estimator) else type(estimator)
        estimator_instances = [None]
        datatypes = [[None]]

    # start yielding checks
    yield from _yield_estimator_checks(estimator_class, estimator_instances, datatypes)

    yield from _yield_soft_dependency_checks(
        estimator_class, estimator_instances, datatypes
    )

    yield from _yield_multithreading_checks(
        estimator_class, estimator_instances, datatypes
    )

    if issubclass(estimator_class, BaseClassifier):
        yield from _yield_classification_checks(
            estimator_class, estimator_instances, datatypes
        )

    if issubclass(estimator_class, BaseEarlyClassifier):
        yield from _yield_early_classification_checks(
            estimator_class, estimator_instances, datatypes
        )

    if issubclass(estimator_class, BaseRegressor):
        yield from _yield_regression_checks(
            estimator_class, estimator_instances, datatypes
        )

    if issubclass(estimator_class, BaseClusterer):
        yield from _yield_clustering_checks(
            estimator_class, estimator_instances, datatypes
        )

    if issubclass(estimator_class, BaseSegmenter):
        yield from _yield_segmentation_checks(
            estimator_class, estimator_instances, datatypes
        )

    if issubclass(estimator_class, BaseAnomalyDetector):
        yield from _yield_anomaly_detection_checks(
            estimator_class, estimator_instances, datatypes
        )

    if issubclass(estimator_class, BaseTransformer):
        yield from _yield_transformation_checks(
            estimator_class, estimator_instances, datatypes
        )


def _yield_estimator_checks(estimator_class, estimator_instances, datatypes):
    """Yield all general checks for an aeon estimator."""
    # only class required
    yield partial(check_create_test_instance, estimator_class=estimator_class)
    yield partial(check_inheritance, estimator_class=estimator_class)
    yield partial(check_has_common_interface, estimator_class=estimator_class)
    yield partial(check_set_params, estimator_class=estimator_class)
    yield partial(check_constructor, estimator_class=estimator_class)
    yield partial(check_estimator_class_tags, estimator_class=estimator_class)

    # test class instances
    for i, estimator in enumerate(estimator_instances):
        # no data needed
        yield partial(check_get_params, estimator=estimator)
        yield partial(check_repr, estimator=estimator)
        yield partial(check_estimator_tags, estimator=estimator)

        if (
            isinstance(estimator, BaseDeepClassifier)
            or isinstance(estimator, BaseDeepRegressor)
            or isinstance(estimator, BaseDeepClusterer)
        ):
            yield partial(check_dl_constructor_initializes_deeply, estimator=estimator)

        # data type irrelevant
        yield partial(
            check_non_state_changing_method,
            estimator=estimator,
            datatype=datatypes[i][0],
        )
        yield partial(
            check_fit_updates_state_and_cloning,
            estimator=estimator,
            datatype=datatypes[i][0],
        )

        if not _get_tag(estimator, "fit_is_empty", default=False):
            yield partial(
                check_raises_not_fitted_error,
                estimator=estimator,
                datatype=datatypes[i][0],
            )

        if not _get_tag(estimator, "cant_pickle", default=False):
            yield partial(
                check_persistence_via_pickle,
                estimator=estimator,
                datatype=datatypes[i][0],
            )

        if not _get_tag(estimator, "non_deterministic", default=False):
            yield partial(
                check_fit_deterministic, estimator=estimator, datatype=datatypes[i][0]
            )


def check_create_test_instance(estimator_class):
    """Check _create_test_instance logic and basic constructor functionality.

    _create_test_instance is the key method used to create test instances in testing.
    If this test does not pass, the validity of the other tests cannot be guaranteed.
    """
    estimator = estimator_class._create_test_instance()

    # Check that method does not construct object of other class than itself
    assert isinstance(estimator, estimator_class), (
        "object returned by _create_test_instance must be an instance of the class, "
        f"found {type(estimator)}"
    )


def check_inheritance(estimator_class):
    """Check that estimator inherits from BaseAeonEstimator."""
    assert issubclass(
        estimator_class, BaseAeonEstimator
    ), f"object {estimator_class} is not a sub-class of BaseAeonEstimator."

    # Usually estimators inherit only from one BaseAeonEstimator type, but in some cases
    # they may inherit both as part of a series/collection split
    n_base_types = sum(
        issubclass(estimator_class, cls) for cls in VALID_ESTIMATOR_BASES.values()
    )
    assert 2 >= n_base_types >= 1, "Estimator should inherit from 1 or 2 base types."

    # Only transformers can inherit from multiple base types currently
    if n_base_types > 1:
        assert issubclass(
            estimator_class, BaseTransformer
        ), "Only transformers can inherit from multiple base types."


def check_has_common_interface(estimator_class):
    """Check estimator implements the common interface."""
    assert issubclass(estimator_class, BaseAeonEstimator)
    assert hasattr(estimator_class, "fit") and callable(estimator_class.fit)
    assert hasattr(estimator_class, "reset") and callable(estimator_class.reset)
    assert hasattr(estimator_class, "clone") and callable(estimator_class.clone)
    assert hasattr(estimator_class, "get_class_tags") and callable(
        estimator_class.get_class_tags
    )
    assert hasattr(estimator_class, "get_class_tag") and callable(
        estimator_class.get_class_tag
    )
    assert hasattr(estimator_class, "get_tags") and callable(estimator_class.get_tags)
    assert hasattr(estimator_class, "get_tag") and callable(estimator_class.get_tag)
    assert hasattr(estimator_class, "set_tags") and callable(estimator_class.set_tags)
    assert hasattr(estimator_class, "get_fitted_params") and callable(
        estimator_class.get_fitted_params
    )

    # axis class parameter is for internal use only
    assert (
        "axis" not in estimator_class.__dict__
    ), "axis should not be a class parameter"

    # Must have at least one set to True
    multi = estimator_class.get_class_tag(tag_name="capability:multivariate")
    uni = estimator_class.get_class_tag(tag_name="capability:univariate")
    assert multi or uni


def check_set_params(estimator_class):
    """Check that set_params works correctly."""
    # some parameters do not have default values, we need to set them
    estimator = estimator_class._create_test_instance()
    required_params_names = [
        p.name
        for p in signature(estimator_class.__init__).parameters.values()
        # dont include self and *args, **kwargs
        if p.name != "self" and p.kind not in [p.VAR_KEYWORD, p.VAR_POSITIONAL]
        # has no default
        and p.default == p.empty
    ]
    params = estimator.get_params()
    init_params = {p: params[p] for p in params if p in required_params_names}

    # default constructed instance except for required parameters
    estimator = estimator_class(**init_params)

    test_params = estimator_class._get_test_params()
    if not isinstance(test_params, list):
        test_params = [test_params]

    for params in test_params:
        # parameter sets may only have parameters that are deviating from defaults
        params_full = estimator.get_params(deep=False)
        params_full.update(params)

        est_after_set = estimator.set_params(**params_full)
        assert (
            est_after_set is estimator
        ), f"set_params of {estimator_class.__name__} does not return self"

        is_equal, equals_msg = deep_equals(
            estimator.get_params(deep=False), params_full, return_msg=True
        )
        msg = (
            f"get_params result of {estimator_class.__name__} (x) does not match "
            f"what was passed to set_params (y). "
            f"Reason for discrepancy: {equals_msg}"
        )
        assert is_equal, msg


def check_constructor(estimator_class):
    """Check that the constructor has sklearn compatible signature and behaviour.

    Tests that:
    * constructor has no varargs
    * tests that constructor constructs an instance of the class
    * tests that all parameters are set in init to an attribute of the same name
    * tests that parameter values are always copied to the attribute and not changed
    * tests that default parameters are a valid type or callable
    """
    assert (
        getfullargspec(estimator_class.__init__).varkw is None
    ), "constructor __init__ should have no varargs"

    estimator = estimator_class._create_test_instance()

    # ensure base class super is called in constructor
    assert hasattr(estimator, "is_fitted"), (
        "Estimator should have an is_fitted attribute after init, if not make sure "
        "you call super().__init__ in the constructor"
    )
    assert (
        estimator.is_fitted is False
    ), "Estimator is_fitted attribute should be set to False after init"
    assert hasattr(estimator, "_tags_dynamic"), (
        "Estimator should have a _tags_dynamic attribute after init, if not make sure "
        "you call super().__init__ in the constructor"
    )
    assert isinstance(
        estimator._tags_dynamic, dict
    ), "Estimator _tags_dynamic attribute should be a dict after init"

    # ensure that each parameter is set in init
    init_params = signature(estimator_class.__init__).parameters
    invalid_attr = set(init_params) - set(vars(estimator)) - {"self"}
    assert not invalid_attr, (
        "Estimator %s should store all parameters"
        " as an attribute during init. Did not find "
        "attributes `%s`." % (estimator.__class__.__name__, sorted(invalid_attr))
    )

    param_values = [
        p
        for p in init_params.values()
        # dont include self and *args, **kwargs
        if p.name != "self" and p.kind not in [p.VAR_KEYWORD, p.VAR_POSITIONAL]
    ]
    required_params_names = [p.name for p in param_values if p.default == p.empty]
    default_value_params = [p for p in param_values if p.default != p.empty]

    params = estimator.get_params()
    init_params = {p: params[p] for p in params if p in required_params_names}

    # default constructed instance except for required parameters
    estimator = estimator_class(**init_params)
    params = estimator.get_params()

    for param in default_value_params:
        allowed_types = {
            str,
            int,
            float,
            bool,
            tuple,
            type(None),
            type,
            np.float64,
            np.int64,
            np.nan,
        }

        assert type(param.default) in allowed_types or callable(param.default), (
            f"Default value of parameter {param.name} is not callable or one of "
            f"the allowed types: {allowed_types}"
        )

        param_value = params[param.name]
        msg = (
            f"Parameter {param.name} was mutated on init. All parameters must be "
            f"stored unchanged."
        )
        if isinstance(param_value, np.ndarray):
            np.testing.assert_array_equal(param_value, param.default, err_msg=msg)
        else:
            if (
                not isinstance(param_value, numbers.Integral)
                and isinstance(param_value, numbers.Real)
                and math.isnan(param_value)
            ):
                # Allows setting default parameters to np.nan
                assert param_value is param.default, msg
            else:
                assert param_value == param.default, msg


def check_estimator_class_tags(estimator_class):
    """Check conventions on estimator tags for class."""
    all_tags = estimator_class.get_class_tags()
    assert isinstance(all_tags, dict)
    assert all(isinstance(key, str) for key in all_tags.keys())

    # check _tags attribute for class
    if hasattr(estimator_class, "_tags"):
        tags = estimator_class._tags
        assert isinstance(tags, dict), (
            f"_tags attribute of {estimator_class} must be dict, "
            f"but found {type(tags)}"
        )
        assert len(tags) > 0, f"_tags dict of class {estimator_class} is empty"
        assert all(isinstance(key, str) for key in tags.keys())

    # validate tags
    check_valid_tags(estimator_class, all_tags)

    # Avoid ambiguous class attributes
    ambiguous_attrs = ("tags", "tags_")
    for attr in ambiguous_attrs:
        assert not hasattr(estimator_class, attr), (
            f"The '{attr}' attribute name is disallowed to avoid confusion with "
            f"estimator tags."
        )

    # Must have at least one set to True
    multi = estimator_class.get_class_tag(tag_name="capability:multivariate")
    uni = estimator_class.get_class_tag(tag_name="capability:univariate")
    assert multi or uni, (
        "Estimator must have at least one of capability:multivariate or "
        "capability:univariate set to True"
    )


def check_get_params(estimator):
    """Check that get_params works correctly."""
    estimator = _clone_estimator(estimator)

    params = estimator.get_params()
    assert isinstance(params, dict)

    shallow_params = estimator.get_params(deep=False)
    deep_params = estimator.get_params(deep=True)

    assert all(item in deep_params.items() for item in shallow_params.items())


def check_repr(estimator):
    """Check that __repr__ call to instance does not raise exceptions."""
    estimator = _clone_estimator(estimator)
    assert isinstance(repr(estimator), str)


def check_estimator_tags(estimator):
    """Check conventions on estimator tags for test objects."""
    estimator = _clone_estimator(estimator)

    all_tags = estimator.get_tags()
    assert isinstance(all_tags, dict)
    assert all(isinstance(key, str) for key in all_tags.keys())

    # check _tags attribute
    if hasattr(estimator, "_tags"):
        assert estimator._tags == estimator.__class__._tags

    # check _tags_dynamic attribute still exists from base
    assert hasattr(estimator, "_tags_dynamic")
    assert isinstance(estimator._tags_dynamic, dict)

    # validate tags
    check_valid_tags(estimator, all_tags)


def check_dl_constructor_initializes_deeply(estimator):
    """Test deep learning estimators pass custom parameters to underlying Network."""
    estimator = _clone_estimator(estimator)

    for key, value in estimator.__dict__.items():
        assert vars(estimator)[key] == value
        # some keys are only relevant to the final model (eg: n_epochs)
        # skip them for the underlying network
        if vars(estimator._network).get(key) is not None:
            assert vars(estimator._network)[key] == value


def check_non_state_changing_method(estimator, datatype):
    """Check that non-state-changing methods behave correctly.

    Non-state-changing methods should not alter the estimator attributes or the
    input arguments. We also check fit does not alter the input arguments here.
    """
    estimator = _clone_estimator(estimator)

    X = deepcopy(FULL_TEST_DATA_DICT[datatype]["train"][0])
    y = deepcopy(FULL_TEST_DATA_DICT[datatype]["train"][1])
    _run_estimator_method(estimator, "fit", datatype, "train")

    assert deep_equals(X, FULL_TEST_DATA_DICT[datatype]["train"][0]) and deep_equals(
        y, FULL_TEST_DATA_DICT[datatype]["train"][1]
    ), f"Estimator: {type(estimator)} has side effects on arguments of fit"

    # dict_before = copy of dictionary of estimator before predict, post fit
    dict_before = estimator.__dict__.copy()
    X = deepcopy(FULL_TEST_DATA_DICT[datatype]["test"][0])
    y = deepcopy(FULL_TEST_DATA_DICT[datatype]["test"][1])

    for method in NON_STATE_CHANGING_METHODS:
        if hasattr(estimator, method) and callable(getattr(estimator, method)):
            _run_estimator_method(estimator, method, datatype, "test")

        assert deep_equals(X, FULL_TEST_DATA_DICT[datatype]["test"][0]) and deep_equals(
            y, FULL_TEST_DATA_DICT[datatype]["test"][1]
        ), f"Estimator: {type(estimator)} has side effects on arguments of {method}"

        # dict_after = dictionary of estimator after predict and fit
        is_equal, msg = deep_equals(estimator.__dict__, dict_before, return_msg=True)
        assert is_equal, (
            f"Estimator: {type(estimator).__name__} changes __dict__ "
            f"during {method}, "
            f"reason/location of discrepancy (x=after, y=before): {msg}"
        )


def check_fit_updates_state_and_cloning(estimator, datatype):
    """Check fit/update state change.

    We test clone here to avoid fitting again in a separate cloning test.

    Tests that:
    * clone returns a new unfitted instance of the estimator
    * fit returns self
    * is_fitted attribute is updated correctly to True after calling fit
    * estimator hyper parameters are not changed in fit
    """
    # do some basic checks for cloning
    estimator_clone = estimator.clone()
    assert isinstance(
        estimator_clone, type(estimator)
    ), "Estimator clone should be of the same type as the original estimator"
    assert (
        estimator_clone is not estimator
    ), "Estimator clone should not be the same object as the original estimator"
    assert (
        estimator_clone.is_fitted is False
    ), "Estimator is_fitted attribute should be set to False after cloning and init"

    # Make a physical copy of the original estimator parameters before fitting.
    estimator = estimator_clone
    original_params = deepcopy(estimator.get_params())

    fitted_estimator = _run_estimator_method(estimator, "fit", datatype, "train")

    # Check fit returns self
    assert (
        fitted_estimator is estimator
    ), f"Estimator: {estimator} does not return self when calling fit"

    # Check is_fitted attribute is updated correctly to True after calling fit
    assert (
        fitted_estimator.is_fitted is True
    ), f"Estimator: {estimator} does not update attribute: is_fitted during fit"

    # Compare the state of the model parameters with the original parameters
    new_params = fitted_estimator.get_params()
    for param_name, original_value in original_params.items():
        new_value = new_params[param_name]

        # We should never change or mutate the internal state of input
        # parameters by default. To check this we use the joblib.hash function
        # that introspects recursively any subobjects to compute a checksum.
        # The only exception to this rule of immutable constructor parameters
        # is possible RandomState instance but in this check we explicitly
        # fixed the random_state params recursively to be integer seeds via clone.
        assert joblib.hash(new_value) == joblib.hash(original_value), (
            "Estimator %s should not change or mutate "
            " the parameter %s from %s to %s during fit."
            % (estimator.__class__.__name__, param_name, original_value, new_value)
        )

    # check that estimator cloned from fitted estimator is not fitted
    estimator_clone = estimator.clone()
    assert (
        estimator_clone.is_fitted is False
    ), "Estimator is_fitted attribute should be set to False after cloning"


def check_raises_not_fitted_error(estimator, datatype):
    """Check exception raised for non-fit method calls to unfitted estimators."""
    import pytest

    estimator = _clone_estimator(estimator)

    for method in NON_STATE_CHANGING_METHODS:
        if hasattr(estimator, method):
            with pytest.raises(NotFittedError, match=r"has not been fitted"):
                _run_estimator_method(estimator, method, datatype, "test")


def _compare_outputs(output1, output2):
    if type(output1) is not type(output2):
        return False
    if isinstance(output1, np.ndarray):
        return np.allclose(output1, output2)
    if isinstance(output1, dict):
        if output1.keys() != output2.keys():
            return False
        for k in output1.keys():
            if not _compare_outputs(output1[k], output2[k]):
                return False
        return True
    if isinstance(output1, tuple):
        if len(output1) != len(output2):
            return False
        for i in range(len(output1)):
            if not _compare_outputs(output1[i], output2[i]):
                return False
        return True
    return False


def check_persistence_via_pickle(estimator, datatype):
    """Check that we can pickle all estimators."""
    estimator = _clone_estimator(estimator, random_state=0)

    _run_estimator_method(estimator, "fit", datatype, "train")

    results = []
    for method in NON_STATE_CHANGING_METHODS_ARRAYLIKE:
        if hasattr(estimator, method) and callable(getattr(estimator, method)):
            output = _run_estimator_method(estimator, method, datatype, "test")
            results.append(output)

    # Serialize and deserialize
    serialized_estimator = pickle.dumps(estimator)
    estimator = pickle.loads(serialized_estimator)

    i = 0
    for method in NON_STATE_CHANGING_METHODS_ARRAYLIKE:
        if hasattr(estimator, method) and callable(getattr(estimator, method)):
            output = _run_estimator_method(estimator, method, datatype, "test")
            assert_array_almost_equal(
                output,
                results[i],
                err_msg=f"Running {method} after fit twice with test "
                f"parameters gives different results.",
            )
            i += 1


def check_fit_deterministic(estimator, datatype):
    """Check that calling fit twice is equivalent to calling it once."""
    estimator = _clone_estimator(estimator, random_state=0)
    _run_estimator_method(estimator, "fit", datatype, "train")

    results = []
    for method in NON_STATE_CHANGING_METHODS_ARRAYLIKE:
        if hasattr(estimator, method) and callable(getattr(estimator, method)):
            output = _run_estimator_method(estimator, method, datatype, "test")
            results.append(output)

    # run fit a second time
    _run_estimator_method(estimator, "fit", datatype, "train")

    # check output of predict/transform etc does not change
    i = 0
    for method in NON_STATE_CHANGING_METHODS_ARRAYLIKE:
        if hasattr(estimator, method) and callable(getattr(estimator, method)):
            output = _run_estimator_method(estimator, method, datatype, "test")
            if not _compare_outputs(output, results[i]):
                raise ValueError(
                    f"Running {method} after fit twice with test "
                    f"parameters gives different results."
                )
            i += 1
