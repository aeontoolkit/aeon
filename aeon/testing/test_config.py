"""Test configuration."""

__maintainer__ = []
__all__ = ["EXCLUDE_ESTIMATORS", "EXCLUDED_TESTS"]

import aeon.testing.utils._cicd_numba_caching  # noqa: F401
from aeon.base import (
    BaseCollectionEstimator,
    BaseEstimator,
    BaseObject,
    BaseSeriesEstimator,
)
from aeon.registry import BASE_CLASS_LIST, BASE_CLASS_LOOKUP, ESTIMATOR_TAG_LIST

# whether to use smaller parameter matrices for test generation and subsample estimators
# per os/version default is False, can be set to True by pytest --prtesting True flag
PR_TESTING = False

# Exclude estimators here for short term fixes
EXCLUDE_ESTIMATORS = [
    # See #2071
    "RandomDilatedShapeletTransform",
    "RDSTClassifier",
    "RDSTRegressor",
]


EXCLUDED_TESTS = {
    # Early classifiers (EC) intentionally retain information from previous predict
    # calls for #1 (test_non_state_changing_method_contract).
    # #2 (test_fit_deterministic), #3 (test_persistence_via_pickle) and #4
    # (test_save_estimators_to_file) are due to predict/predict_proba returning two
    # items and that breaking assert_array_equal.
    "TEASER": [  # EC
        "test_non_state_changing_method_contract",
        "test_fit_deterministic",
        "test_persistence_via_pickle",
        "test_save_estimators_to_file",
    ],
    "ProbabilityThresholdEarlyClassifier": [  # EC
        "test_non_state_changing_method_contract",
        "test_fit_deterministic",
        "test_persistence_via_pickle",
        "test_save_estimators_to_file",
    ],
    # has a keras fail, unknown reason, see #1387
    "LearningShapeletClassifier": ["check_fit_deterministic"],
    # does not fit structure for test, needs investigation
    "TapNetClassifier": ["check_classifier_random_state_deep_learning"],
    "TapNetRegressor": ["check_regressor_random_state_deep_learning"],
    # needs investigation
    "SASTClassifier": ["check_fit_deterministic"],
    "RSASTClassifier": ["check_fit_deterministic"],
    "AEFCNClusterer": ["check_fit_updates_state"],
    "AEResNetClusterer": ["check_fit_updates_state"],
}

# We use estimator tags in addition to class hierarchies to further distinguish
# estimators into different categories. This is useful for defining and running
# common tests for estimators with the same tags.
VALID_ESTIMATOR_TAGS = tuple(ESTIMATOR_TAG_LIST)

# NON_STATE_CHANGING_METHODS =
# methods that should not change the state of the estimator, that is, they should
# not change fitted parameters or hyper-parameters. They are also the methods that
# "apply" the fitted estimator to data and useful for checking results.
# NON_STATE_CHANGING_METHODS_ARRAYLIK =
# non-state-changing methods that return an array-like output

NON_STATE_CHANGING_METHODS_ARRAYLIKE = (
    "predict",
    "predict_var",
    "predict_proba",
    "decision_function",
    "transform",
)

NON_STATE_CHANGING_METHODS = NON_STATE_CHANGING_METHODS_ARRAYLIKE + (
    "get_fitted_params",
)

# The following gives a list of valid estimator base classes.
CORE_BASE_TYPES = (
    BaseEstimator,
    BaseObject,
    BaseCollectionEstimator,
    BaseSeriesEstimator,
)
VALID_ESTIMATOR_BASE_TYPES = tuple(set(BASE_CLASS_LIST).difference(CORE_BASE_TYPES))

VALID_ESTIMATOR_TYPES = (
    BaseEstimator,
    *VALID_ESTIMATOR_BASE_TYPES,
)

VALID_ESTIMATOR_BASE_TYPE_LOOKUP = BASE_CLASS_LOOKUP
