"""Tests for all estimators."""

import numbers
import pickle
import types
from copy import deepcopy
from functools import partial
from inspect import getfullargspec, isclass, signature

import joblib
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_get_params_invariance

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.anomaly_detection.whole_series.base import BaseCollectionAnomalyDetector
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
from aeon.similarity_search import BaseSimilaritySearch
from aeon.testing.estimator_checking._yield_anomaly_detection_checks import (
    _yield_anomaly_detection_checks,
)
from aeon.testing.estimator_checking._yield_classification_checks import (
    _yield_classification_checks,
)
from aeon.testing.estimator_checking._yield_clustering_checks import (
    _yield_clustering_checks,
)
from aeon.testing.estimator_checking._yield_collection_anomaly_detection_checks import (
    _yield_collection_anomaly_detection_checks,
)
from aeon.testing.estimator_checking._yield_collection_transformation_checks import (
    _yield_collection_transformation_checks,
)
from aeon.testing.estimator_checking._yield_early_classification_checks import (
    _yield_early_classification_checks,
)
from aeon.testing.estimator_checking._yield_regression_checks import (
    _yield_regression_checks,
)
from aeon.testing.estimator_checking._yield_segmentation_checks import (
    _yield_segmentation_checks,
)
from aeon.testing.estimator_checking._yield_series_transformation_checks import (
    _yield_series_transformation_checks,
)
from aeon.testing.estimator_checking._yield_similarity_search_checks import (
    _yield_similarity_search_checks,
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
from aeon.testing.utils.estimator_checks import (
    _assert_array_almost_equal,
    _get_args,
    _get_tag,
    _list_required_methods,
    _run_estimator_method,
)
from aeon.transformations.base import BaseTransformer
from aeon.transformations.collection import BaseCollectionTransformer
from aeon.transformations.series import BaseSeriesTransformer
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

    if issubclass(estimator_class, BaseCollectionAnomalyDetector):
        yield from _yield_collection_anomaly_detection_checks(
            estimator_class, estimator_instances, datatypes
        )

    if issubclass(estimator_class, BaseSimilaritySearch):
        yield from _yield_similarity_search_checks(
            estimator_class, estimator_instances, datatypes
        )

    if issubclass(estimator_class, BaseTransformer):
        yield from _yield_transformation_checks(
            estimator_class, estimator_instances, datatypes
        )

    if issubclass(estimator_class, BaseCollectionTransformer):
        yield from _yield_collection_transformation_checks(
            estimator_class, estimator_instances, datatypes
        )

    if issubclass(estimator_class, BaseSeriesTransformer):
        yield from _yield_series_transformation_checks(
            estimator_class, estimator_instances, datatypes
        )


def _yield_estimator_checks(estimator_class, estimator_instances, datatypes):
    """Yield all general checks for an aeon estimator."""
    # only class required
    yield partial(check_create_test_instance, estimator_class=estimator_class)
    yield partial(check_inheritance, estimator_class=estimator_class)
    yield partial(check_has_common_interface, estimator_class=estimator_class)
    yield partial(check_set_params_sklearn, estimator_class=estimator_class)
    yield partial(check_constructor, estimator_class=estimator_class)
    yield partial(check_estimator_class_tags, estimator_class=estimator_class)

    # test class instances
    for i, estimator in enumerate(estimator_instances):
        # no data needed
        yield partial(check_get_params, estimator=estimator)
        yield partial(check_set_params, estimator=estimator)
        yield partial(check_clone, estimator=estimator)
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
            check_fit_updates_state, estimator=estimator, datatype=datatypes[i][0]
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

    Also tests inheritance and super call logic in the constructor.

    Tests that:
    * _create_test_instance results in an instance of estimator_class
    * __init__ calls super.__init__
    * _tags_dynamic attribute for tag inspection is present after construction
    """
    estimator = estimator_class._create_test_instance()

    # Check that method does not construct object of other class than itself
    assert isinstance(estimator, estimator_class), (
        "object returned by _create_test_instance must be an instance of the class, "
        f"found {type(estimator)}"
    )

    msg = (
        f"{estimator_class.__name__}.__init__ should call super().__init__, the "
        "estimator does not produce the attributes this call would produce."
    )
    assert hasattr(estimator, "_tags_dynamic"), msg


# todo consider removing the multiple base class allowance.
def check_inheritance(estimator_class):
    """Check that estimator inherits from BaseAeonEstimator."""
    assert issubclass(
        estimator_class, BaseAeonEstimator
    ), f"object {estimator_class} is not a sub-class of BaseAeonEstimator."

    if hasattr(estimator_class, "fit"):
        assert issubclass(estimator_class, BaseAeonEstimator), (
            f"estimator: {estimator_class} has fit method, but"
            f"is not a sub-class of BaseAeonEstimator."
        )

    # Usually estimators inherit only from one BaseAeonEstimator type, but in some cases
    # they may be predictor and transformer at the same time (e.g. pipelines)
    n_base_types = sum(
        issubclass(estimator_class, cls) for cls in VALID_ESTIMATOR_BASES.values()
    )

    assert 2 >= n_base_types >= 1

    # If the estimator inherits from more than one base estimator type, we check if
    # one of them is a transformer base type
    if n_base_types > 1:
        assert issubclass(estimator_class, BaseTransformer)


def check_has_common_interface(estimator_class):
    """Check estimator implements the common interface."""
    # Check class for type of attribute
    if isinstance(estimator_class, BaseAeonEstimator):
        assert isinstance(estimator_class.is_fitted, property)

    required_methods = _list_required_methods(estimator_class)

    for attr in required_methods:
        assert hasattr(
            estimator_class, attr
        ), f"Estimator: {estimator_class.__name__} does not implement attribute: {attr}"

    if hasattr(estimator_class, "inverse_transform"):
        assert hasattr(estimator_class, "transform")
    if hasattr(estimator_class, "predict_proba"):
        assert hasattr(estimator_class, "predict")


def check_set_params_sklearn(estimator_class):
    """Check that set_params works correctly, mirrors sklearn check_set_params.

    Instead of the "fuzz values" in sklearn's check_set_params,
    we use the other test parameter settings (which are assumed valid).
    This guarantees settings which play along with the __init__ content.
    """
    estimator = estimator_class._create_test_instance()
    test_params = estimator_class._get_test_params()
    if not isinstance(test_params, list):
        test_params = [test_params]

    for params in test_params:
        # we construct the full parameter set for params
        # params may only have parameters that are deviating from defaults
        # in order to set non-default parameters back to defaults
        params_full = estimator.get_params(deep=False)
        params_full.update(params)

        msg = f"set_params of {estimator_class.__name__} does not return self"
        est_after_set = estimator.set_params(**params_full)
        assert est_after_set is estimator, msg

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

    Based on sklearn check_estimator testing of __init__ logic.
    Uses _create_test_instance to create an instance.
    Assumes test_create_test_instance has passed and certified _create_test_instance.

    Tests that:
    * constructor has no varargs
    * tests that constructor constructs an instance of the class
    * tests that all parameters are set in init to an attribute of the same name
    * tests that parameter values are always copied to the attribute and not changed
    * tests that default parameters are one of the following:
        None, str, int, float, bool, tuple, function, joblib memory, numpy primitive
        (other type parameters should be None, default handling should be by writing
        the default to attribute of a different name, e.g., my_param_ not my_param)
    """
    msg = "constructor __init__ should have no varargs"
    assert getfullargspec(estimator_class.__init__).varkw is None, msg

    estimator = estimator_class._create_test_instance()
    assert isinstance(estimator, estimator_class)

    # Ensure that each parameter is set in init
    init_params = _get_args(type(estimator).__init__)
    invalid_attr = set(init_params) - set(vars(estimator)) - {"self"}
    assert not invalid_attr, (
        "Estimator %s should store all parameters"
        " as an attribute during init. Did not find "
        "attributes `%s`." % (estimator.__class__.__name__, sorted(invalid_attr))
    )

    # Ensure that init does nothing but set parameters
    # No logic/interaction with other parameters
    def param_filter(p):
        """Identify hyper parameters of an estimator."""
        return p.name != "self" and p.kind not in [p.VAR_KEYWORD, p.VAR_POSITIONAL]

    init_params = [
        p for p in signature(estimator.__init__).parameters.values() if param_filter(p)
    ]

    params = estimator.get_params()

    test_params = estimator_class._get_test_params()
    if isinstance(test_params, list):
        test_params = test_params[0]
    test_params = test_params.keys()

    init_params = [param for param in init_params if param.name not in test_params]

    for param in init_params:
        assert param.default != param.empty, (
            "parameter `%s` for %s has no default value and is not "
            "set in _get_test_params" % (param.name, estimator.__class__.__name__)
        )
        if type(param.default) is type:
            assert param.default in [np.float64, np.int64]
        else:
            assert type(param.default) in [
                str,
                int,
                float,
                bool,
                tuple,
                type(None),
                np.float64,
                types.FunctionType,
                joblib.Memory,
            ]

        param_value = params[param.name]
        if isinstance(param_value, np.ndarray):
            np.testing.assert_array_equal(param_value, param.default)
        else:
            if bool(isinstance(param_value, numbers.Real) and np.isnan(param_value)):
                # Allows to set default parameters to np.nan
                assert param_value is param.default, param.name
            else:
                assert param_value == param.default, param.name


def check_estimator_class_tags(estimator_class):
    """Check conventions on estimator tags for class."""
    # check get_class_tags method is retained from base
    assert hasattr(estimator_class, "get_class_tags")
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


def check_get_params(estimator):
    """Check that get_params works correctly."""
    params = estimator.get_params()
    assert isinstance(params, dict)
    check_get_params_invariance(estimator.__class__.__name__, estimator)


def check_set_params(estimator):
    """Check that set_params works correctly."""
    estimator = _clone_estimator(estimator)
    params = estimator.get_params()

    msg = f"set_params of {type(estimator).__name__} does not return self"
    assert estimator.set_params(**params) is estimator, msg

    is_equal, equals_msg = deep_equals(estimator.get_params(), params, return_msg=True)
    msg = (
        f"get_params result of {type(estimator).__name__} (x) does not match "
        f"what was passed to set_params (y). Reason for discrepancy: {equals_msg}"
    )
    assert is_equal, msg


def check_clone(estimator):
    """Check that clone method does not raise exceptions and results in a clone.

    A clone of an object x is an object that:
    * has same class and parameters as x
    * is not identical with x
     * is unfitted (even if x was fitted)
    """
    est_clone = estimator.clone()
    assert isinstance(est_clone, type(estimator))
    assert est_clone is not estimator
    if hasattr(est_clone, "is_fitted"):
        assert not est_clone.is_fitted


# todo roll into another test
def check_repr(estimator):
    """Check that __repr__ call to instance does not raise exceptions."""
    repr(estimator)


def check_estimator_tags(estimator):
    """Check conventions on estimator tags for test objects."""
    # check get_tags method is retained from base
    assert hasattr(estimator, "get_tags")
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
    """Test DL estimators that they pass custom parameters to underlying Network."""
    for key, value in estimator.__dict__.items():
        assert vars(estimator)[key] == value
        # some keys are only relevant to the final model (eg: n_epochs)
        # skip them for the underlying network
        if vars(estimator._network).get(key) is not None:
            assert vars(estimator._network)[key] == value


def check_non_state_changing_method(estimator, datatype):
    """Check that non-state-changing methods behave as per interface contract.

    Check the following contract on non-state-changing methods:
    1. do not change state of the estimator, i.e., any attributes
        (including hyper-parameters and fitted parameters)
    2. expected output type of the method matches actual output type
        - only for abstract BaseAeonEstimator methods, common to all estimators.
        List of BaseAeonEstimator methods tested: get_fitted_params
        Subclass specific method outputs are tested in TestAll[estimatortype] class
    3. the state of method arguments does not change
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
        dict_after = estimator.__dict__
        is_equal, msg = deep_equals(dict_after, dict_before, return_msg=True)
        assert is_equal, (
            f"Estimator: {type(estimator).__name__} changes __dict__ "
            f"during {method}, "
            f"reason/location of discrepancy (x=after, y=before): {msg}"
        )


def check_fit_updates_state(estimator, datatype):
    """Check fit/update state change.

    1. Check estimator_instance calls base class constructor
    2. Check is_fitted attribute is set correctly to False before fit, at init
        This is testing base class functionality, but its fast
    3. Check fit returns self
    4. Check is_fitted attribute is updated correctly to True after calling fit
    5. Check estimator hyper parameters are not changed in fit
    """
    # Check that fit updates the is-fitted states
    estimator = _clone_estimator(estimator)

    msg = (
        f"{type(estimator).__name__}.__init__ should call "
        f"super({type(estimator).__name__}, self).__init__, "
        "but that does not seem to be the case. Please ensure to call the "
        f"parent class's constructor in {type(estimator).__name__}.__init__"
    )
    assert hasattr(estimator, "is_fitted"), msg

    # Check is_fitted attribute is set correctly to False before fit, at init
    assert (
        not estimator.is_fitted
    ), f"Estimator: {estimator} does not initiate attribute: is_fitted to False"

    # Make a physical copy of the original estimator parameters before fitting.
    original_params = deepcopy(estimator.get_params())

    fitted_estimator = _run_estimator_method(estimator, "fit", datatype, "train")

    # Check fit returns self
    assert (
        fitted_estimator is estimator
    ), f"Estimator: {estimator} does not return self when calling fit"

    # Check is_fitted attribute is updated correctly to True after calling fit
    assert (
        fitted_estimator.is_fitted
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
        # fixed the random_state params recursively to be integer seeds.
        assert joblib.hash(new_value) == joblib.hash(original_value), (
            "Estimator %s should not change or mutate "
            " the parameter %s from %s to %s during fit."
            % (estimator.__class__.__name__, param_name, original_value, new_value)
        )


def check_raises_not_fitted_error(estimator, datatype):
    """Check exception raised for non-fit method calls to unfitted estimators.

    Tries to run all methods in NON_STATE_CHANGING_METHODS with valid scenario,
    but before fit has been called on the estimator.

    This should raise a NotFittedError if correctly caught,
    normally by a self.check_is_fitted() call in the method's boilerplate.

    Raises
    ------
    Exception if NotFittedError is not raised by non-state changing method
    """
    # call methods without prior fitting and check that they raise NotFittedError
    for method in NON_STATE_CHANGING_METHODS:
        if hasattr(estimator, method):
            with pytest.raises(NotFittedError, match=r"has not been fitted"):
                _run_estimator_method(estimator, method, datatype, "test")


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

            _assert_array_almost_equal(
                output,
                results[i],
                err_msg=f"Running {method} after fit twice with test "
                f"parameters gives different results.",
            )

            i += 1


def check_fit_deterministic(estimator, datatype):
    """Test that fit is deterministic.

    Check that calling fit twice is equivalent to calling it once.
    """
    estimator = _clone_estimator(estimator, random_state=0)
    _run_estimator_method(estimator, "fit", datatype, "train")

    results = []
    for method in NON_STATE_CHANGING_METHODS_ARRAYLIKE:
        if hasattr(estimator, method) and callable(getattr(estimator, method)):
            output = _run_estimator_method(estimator, method, datatype, "test")
            results.append(output)

    # run fit and other methods a second time
    _run_estimator_method(estimator, "fit", datatype, "train")

    i = 0
    for method in NON_STATE_CHANGING_METHODS_ARRAYLIKE:
        if hasattr(estimator, method) and callable(getattr(estimator, method)):
            output = _run_estimator_method(estimator, method, datatype, "test")

            _assert_array_almost_equal(
                output,
                results[i],
                err_msg=f"Running {method} after fit twice with test "
                f"parameters gives different results.",
            )

            i += 1


# def check_multiprocessing_idempotent(estimator):
#     """Test that single and multi-process run results are identical.
#
#     Check that running an estimator on a single process is no different to running
#     it on multiple processes. We also check that we can set n_jobs=-1 to make use
#     of all CPUs. The test is not really necessary though, as we rely on joblib for
#     parallelization and can trust that it works as expected.
#     """
#     method_nsc = method_nsc_arraylike
#     params = estimator_instance.get_params()
#
#     if "n_jobs" in params:
#         # run on a single process
#         # -----------------------
#         estimator = deepcopy(estimator_instance)
#         estimator.set_params(n_jobs=1)
#         set_random_state(estimator)
#         result_single_process = scenario.run(
#             estimator, method_sequence=["fit", method_nsc]
#         )
#
#         # run on multiple processes
#         # -------------------------
#         estimator = deepcopy(estimator_instance)
#         estimator.set_params(n_jobs=-1)
#         set_random_state(estimator)
#         result_multiple_process = scenario.run(
#             estimator, method_sequence=["fit", method_nsc]
#         )
#         _assert_array_equal(
#             result_single_process,
#             result_multiple_process,
#             err_msg="Results are not equal for n_jobs=1 and n_jobs=-1",
#         )
