"""Datasets used for testing."""

import numpy as np

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.anomaly_detection.whole_series.base import BaseCollectionAnomalyDetector
from aeon.base import BaseCollectionEstimator, BaseSeriesEstimator
from aeon.classification import BaseClassifier
from aeon.classification.early_classification import BaseEarlyClassifier
from aeon.clustering import BaseClusterer
from aeon.regression import BaseRegressor
from aeon.segmentation import BaseSegmenter
from aeon.similarity_search import BaseSimilaritySearch
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_dataframe_collection,
    make_example_2d_numpy_collection,
    make_example_2d_numpy_series,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
    make_example_dataframe_list,
    make_example_multi_index_dataframe,
)
from aeon.transformations.collection import BaseCollectionTransformer
from aeon.transformations.series import BaseSeriesTransformer

data_rng = np.random.RandomState(42)


EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION = {
    "numpy3D": {
        "train": make_example_3d_numpy(
            n_cases=10,
            n_channels=1,
            n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
        "test": make_example_3d_numpy(
            n_cases=5,
            n_channels=1,
            n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
    },
    "numpy2D": {
        "train": make_example_2d_numpy_collection(
            n_cases=10,
            n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
        "test": make_example_2d_numpy_collection(
            n_cases=5,
            n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
    },
    "np-list": {
        "train": make_example_3d_numpy_list(
            n_cases=10,
            n_channels=1,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
        "test": make_example_3d_numpy_list(
            n_cases=5,
            n_channels=1,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
    },
    "df-list": {
        "train": make_example_dataframe_list(
            n_cases=10,
            n_channels=1,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
        "test": make_example_dataframe_list(
            n_cases=5,
            n_channels=1,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
    },
    "pd-wide": {
        "train": make_example_2d_dataframe_collection(
            n_cases=10,
            n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
        "test": make_example_2d_dataframe_collection(
            n_cases=5,
            n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
    },
    "pd-multiindex": {
        "train": make_example_multi_index_dataframe(
            n_cases=10,
            n_channels=1,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
        "test": make_example_multi_index_dataframe(
            n_cases=5,
            n_channels=1,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
    },
}

EQUAL_LENGTH_UNIVARIATE_REGRESSION = {
    "numpy3D": {
        "train": make_example_3d_numpy(
            n_cases=10,
            n_channels=1,
            n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
        "test": make_example_3d_numpy(
            n_cases=5,
            n_channels=1,
            n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
    },
    "numpy2D": {
        "train": make_example_2d_numpy_collection(
            n_cases=10,
            n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
        "test": make_example_2d_numpy_collection(
            n_cases=5,
            n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
    },
    "np-list": {
        "train": make_example_3d_numpy_list(
            n_cases=10,
            n_channels=1,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
        "test": make_example_3d_numpy_list(
            n_cases=5,
            n_channels=1,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
    },
    "df-list": {
        "train": make_example_dataframe_list(
            n_cases=10,
            n_channels=1,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
        "test": make_example_dataframe_list(
            n_cases=5,
            n_channels=1,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
    },
    "pd-wide": {
        "train": make_example_2d_dataframe_collection(
            n_cases=10,
            n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
        "test": make_example_2d_dataframe_collection(
            n_cases=5,
            n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
    },
    "pd-multiindex": {
        "train": make_example_multi_index_dataframe(
            n_cases=10,
            n_channels=1,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
        "test": make_example_multi_index_dataframe(
            n_cases=5,
            n_channels=1,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
    },
}

EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION = {
    "numpy3D": {
        "train": make_example_3d_numpy(
            n_cases=10,
            n_channels=2,
            n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
        "test": make_example_3d_numpy(
            n_cases=5,
            n_channels=2,
            n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
    },
    "np-list": {
        "train": make_example_3d_numpy_list(
            n_cases=10,
            n_channels=2,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
        "test": make_example_3d_numpy_list(
            n_cases=5,
            n_channels=2,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
    },
    "df-list": {
        "train": make_example_dataframe_list(
            n_cases=10,
            n_channels=2,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
        "test": make_example_dataframe_list(
            n_cases=5,
            n_channels=2,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
    },
    "pd-multiindex": {
        "train": make_example_multi_index_dataframe(
            n_cases=10,
            n_channels=2,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
        "test": make_example_multi_index_dataframe(
            n_cases=5,
            n_channels=2,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
    },
}

EQUAL_LENGTH_MULTIVARIATE_REGRESSION = {
    "numpy3D": {
        "train": make_example_3d_numpy(
            n_cases=10,
            n_channels=2,
            n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
        "test": make_example_3d_numpy(
            n_cases=5,
            n_channels=2,
            n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
    },
    "np-list": {
        "train": make_example_3d_numpy_list(
            n_cases=10,
            n_channels=2,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
        "test": make_example_3d_numpy_list(
            n_cases=5,
            n_channels=2,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
    },
    "df-list": {
        "train": make_example_dataframe_list(
            n_cases=10,
            n_channels=2,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
        "test": make_example_dataframe_list(
            n_cases=5,
            n_channels=2,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
    },
    "pd-multiindex": {
        "train": make_example_multi_index_dataframe(
            n_cases=10,
            n_channels=2,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
        "test": make_example_multi_index_dataframe(
            n_cases=5,
            n_channels=2,
            min_n_timepoints=20,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
    },
}

UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION = {
    "np-list": {
        "train": make_example_3d_numpy_list(
            n_cases=10,
            n_channels=1,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
        "test": make_example_3d_numpy_list(
            n_cases=5,
            n_channels=1,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
    },
    "df-list": {
        "train": make_example_dataframe_list(
            n_cases=10,
            n_channels=1,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
        "test": make_example_dataframe_list(
            n_cases=5,
            n_channels=1,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
    },
    "pd-multiindex": {
        "train": make_example_multi_index_dataframe(
            n_cases=10,
            n_channels=1,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
        "test": make_example_multi_index_dataframe(
            n_cases=5,
            n_channels=1,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
    },
}

UNEQUAL_LENGTH_UNIVARIATE_REGRESSION = {
    "np-list": {
        "train": make_example_3d_numpy_list(
            n_cases=10,
            n_channels=1,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
        "test": make_example_3d_numpy_list(
            n_cases=5,
            n_channels=1,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
    },
    "df-list": {
        "train": make_example_dataframe_list(
            n_cases=10,
            n_channels=1,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
        "test": make_example_dataframe_list(
            n_cases=5,
            n_channels=1,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
    },
    "pd-multiindex": {
        "train": make_example_multi_index_dataframe(
            n_cases=10,
            n_channels=1,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
        "test": make_example_multi_index_dataframe(
            n_cases=5,
            n_channels=1,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
    },
}

UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION = {
    "np-list": {
        "train": make_example_3d_numpy_list(
            n_cases=10,
            n_channels=2,
            max_n_timepoints=20,
            min_n_timepoints=10,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
        "test": make_example_3d_numpy_list(
            n_cases=5,
            n_channels=2,
            max_n_timepoints=20,
            min_n_timepoints=10,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
    },
    "df-list": {
        "train": make_example_dataframe_list(
            n_cases=10,
            n_channels=2,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
        "test": make_example_dataframe_list(
            n_cases=5,
            n_channels=2,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
    },
    "pd-multiindex": {
        "train": make_example_multi_index_dataframe(
            n_cases=10,
            n_channels=2,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
        "test": make_example_multi_index_dataframe(
            n_cases=5,
            n_channels=2,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
        ),
    },
}

UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION = {
    "np-list": {
        "train": make_example_3d_numpy_list(
            n_cases=10,
            n_channels=2,
            max_n_timepoints=20,
            min_n_timepoints=10,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
        "test": make_example_3d_numpy_list(
            n_cases=5,
            n_channels=2,
            max_n_timepoints=20,
            min_n_timepoints=10,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
    },
    "df-list": {
        "train": make_example_dataframe_list(
            n_cases=10,
            n_channels=2,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
        "test": make_example_dataframe_list(
            n_cases=5,
            n_channels=2,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
    },
    "pd-multiindex": {
        "train": make_example_multi_index_dataframe(
            n_cases=10,
            n_channels=2,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
        "test": make_example_multi_index_dataframe(
            n_cases=5,
            n_channels=2,
            min_n_timepoints=10,
            max_n_timepoints=20,
            random_state=data_rng.randint(np.iinfo(np.int32).max),
            regression_target=True,
        ),
    },
}

X_classification_missing_train, y_classification_missing_train = make_example_3d_numpy(
    n_cases=10,
    n_channels=1,
    n_timepoints=20,
    random_state=data_rng.randint(np.iinfo(np.int32).max),
)
X_classification_missing_test, y_classification_missing_test = make_example_3d_numpy(
    n_cases=5,
    n_channels=1,
    n_timepoints=20,
    random_state=data_rng.randint(np.iinfo(np.int32).max),
)
X_classification_missing_train[:, :, data_rng.choice(20, 2)] = np.nan
X_classification_missing_test[:, :, data_rng.choice(20, 2)] = np.nan

MISSING_VALUES_CLASSIFICATION = {
    "numpy3D": {
        "train": (X_classification_missing_train, y_classification_missing_train),
        "test": (X_classification_missing_test, y_classification_missing_test),
    }
}

X_classification_missing_train, y_classification_missing_train = make_example_3d_numpy(
    n_cases=10,
    n_channels=1,
    n_timepoints=20,
    random_state=data_rng.randint(np.iinfo(np.int32).max),
    regression_target=True,
)
X_classification_missing_test, y_classification_missing_test = make_example_3d_numpy(
    n_cases=5,
    n_channels=1,
    n_timepoints=20,
    random_state=data_rng.randint(np.iinfo(np.int32).max),
    regression_target=True,
)
X_classification_missing_train[:, :, data_rng.choice(20, 2)] = np.nan
X_classification_missing_test[:, :, data_rng.choice(20, 2)] = np.nan

MISSING_VALUES_REGRESSION = {
    "numpy3D": {
        "train": (X_classification_missing_train, y_classification_missing_train),
        "test": (X_classification_missing_test, y_classification_missing_test),
    }
}

X_series = make_example_1d_numpy(
    n_timepoints=40, random_state=data_rng.randint(np.iinfo(np.int32).max)
)
X_series2 = X_series[20:40]
X_series = X_series[:20]
UNIVARIATE_SERIES_NOLABEL = {"train": (X_series, None), "test": (X_series2, None)}

X_series_mv = make_example_2d_numpy_series(
    n_timepoints=40,
    n_channels=2,
    axis=1,
    random_state=data_rng.randint(np.iinfo(np.int32).max),
)
X_series_mv2 = X_series_mv[:, 20:40]
X_series_mv = X_series_mv[:, :20]
MULTIVARIATE_SERIES_NOLABEL = {
    "train": (X_series_mv, None),
    "test": (X_series_mv2, None),
}

X_series_mi = make_example_1d_numpy(
    n_timepoints=40, random_state=data_rng.randint(np.iinfo(np.int32).max)
)
X_series_mi2 = X_series_mi[20:40]
X_series_mi2[data_rng.choice(20, 1)] = np.nan
X_series_mi = X_series_mi[:20]
X_series_mi[data_rng.choice(20, 2)] = np.nan
MISSING_VALUES_NOLABEL = {"train": (X_series_mi, None), "test": (X_series_mi2, None)}

FULL_TEST_DATA_DICT = {}
# Collection
FULL_TEST_DATA_DICT.update(
    {
        f"EqualLengthUnivariate-Classification-{k}": v
        for k, v in EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION.items()
    }
)
FULL_TEST_DATA_DICT.update(
    {
        f"EqualLengthUnivariate-Regression-{k}": v
        for k, v in EQUAL_LENGTH_UNIVARIATE_REGRESSION.items()
    }
)
FULL_TEST_DATA_DICT.update(
    {
        f"EqualLengthMultivariate-Classification-{k}": v
        for k, v in EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION.items()
    }
)
FULL_TEST_DATA_DICT.update(
    {
        f"EqualLengthMultivariate-Regression-{k}": v
        for k, v in EQUAL_LENGTH_MULTIVARIATE_REGRESSION.items()
    }
)
FULL_TEST_DATA_DICT.update(
    {
        f"UnequalLengthUnivariate-Classification-{k}": v
        for k, v in UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION.items()
    }
)
FULL_TEST_DATA_DICT.update(
    {
        f"UnequalLengthUnivariate-Regression-{k}": v
        for k, v in UNEQUAL_LENGTH_UNIVARIATE_REGRESSION.items()
    }
)
FULL_TEST_DATA_DICT.update(
    {
        f"UnequalLengthMultivariate-Classification-{k}": v
        for k, v in UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION.items()
    }
)
FULL_TEST_DATA_DICT.update(
    {
        f"UnequalLengthMultivariate-Regression-{k}": v
        for k, v in UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION.items()
    }
)
FULL_TEST_DATA_DICT.update(
    {
        f"MissingValues-Classification-{k}": v
        for k, v in MISSING_VALUES_CLASSIFICATION.items()
    }
)
FULL_TEST_DATA_DICT.update(
    {f"MissingValues-Regression-{k}": v for k, v in MISSING_VALUES_REGRESSION.items()}
)
# Series
FULL_TEST_DATA_DICT.update({"UnivariateSeries-NoLabel": UNIVARIATE_SERIES_NOLABEL})
FULL_TEST_DATA_DICT.update({"MultivariateSeries-NoLabel": MULTIVARIATE_SERIES_NOLABEL})
FULL_TEST_DATA_DICT.update({"MissingValues-NoLabel": MISSING_VALUES_NOLABEL})


def _get_datatypes_for_estimator(estimator):
    """Get all data types for estimator.

    Parameters
    ----------
    estimator : BaseAeonEstimator instance or class
        Estimator instance or class to check for valid input data types.

    Returns
    -------
    datatypes : list of tuple
        List of valid data types keys for the estimator usable in FULL_TEST_DATA_DICT
        and TEST_LABEL_DICT. Each tuple is formatted (data_key, label_key).
    """
    datatypes = []
    univariate, multivariate, unequal_length, missing_values = (
        _get_capabilities_for_estimator(estimator)
    )
    label_type = _get_label_type_for_estimator(estimator)

    inner_types = estimator.get_tag("X_inner_type")
    if not isinstance(inner_types, list):
        inner_types = [inner_types]

    if isinstance(estimator, BaseCollectionEstimator):
        for inner_type in inner_types:
            if univariate:
                s = f"EqualLengthUnivariate-{label_type}-{inner_type}"
                if s in FULL_TEST_DATA_DICT:
                    datatypes.append(s)

                if unequal_length:
                    s = f"UnequalLengthUnivariate-{label_type}-{inner_type}"
                    if s in FULL_TEST_DATA_DICT:
                        datatypes.append(s)

            if multivariate:
                s = f"EqualLengthMultivariate-{label_type}-{inner_type}"
                if s in FULL_TEST_DATA_DICT:
                    datatypes.append(s)

                if unequal_length:
                    s = f"UnequalLengthMultivariate-{label_type}-{inner_type}"
                    if s in FULL_TEST_DATA_DICT:
                        datatypes.append(s)

        if missing_values:
            datatypes.append(f"MissingValues-{label_type}-numpy3D")
    elif isinstance(estimator, BaseSeriesEstimator):
        if univariate:
            datatypes.append("UnivariateSeries-NoLabel")
        if multivariate:
            datatypes.append("MultivariateSeries-NoLabel")
        if missing_values:
            datatypes.append("MissingValues-NoLabel")
    else:
        raise ValueError(f"Unknown estimator type: {type(estimator)}")

    if len(datatypes) == 0:
        raise ValueError(f"No valid data types found for estimator {estimator}")

    return datatypes


def _get_capabilities_for_estimator(estimator):
    """Get capabilities for estimator.

    Parameters
    ----------
    estimator : BaseAeonEstimator instance or class
        Estimator instance or class to check for valid input data types.

    Returns
    -------
    capabilities : tuple of bool
        Tuple of valid capabilities for the estimator.
    """
    univariate = estimator.get_tag(
        "capability:univariate", raise_error=False, tag_value_default=True
    )
    multivariate = estimator.get_tag(
        "capability:multivariate", raise_error=False, tag_value_default=False
    )
    unequal_length = estimator.get_tag(
        "capability:unequal_length", raise_error=False, tag_value_default=False
    )
    missing_values = estimator.get_tag(
        "capability:missing_values", raise_error=False, tag_value_default=False
    )
    return univariate, multivariate, unequal_length, missing_values


def _get_label_type_for_estimator(estimator):
    """Get label type for estimator.

    Parameters
    ----------
    estimator : BaseAeonEstimator instance or class
        Estimator instance or class to check for valid input data types.

    Returns
    -------
    label_type : str
        Label type key for the estimator for use in TEST_LABEL_DICT.
    """
    if (
        isinstance(estimator, BaseClassifier)
        or isinstance(estimator, BaseEarlyClassifier)
        or isinstance(estimator, BaseClusterer)
        or isinstance(estimator, BaseCollectionTransformer)
        or isinstance(estimator, BaseSimilaritySearch)
        or isinstance(estimator, BaseCollectionAnomalyDetector)
    ):
        label_type = "Classification"
    elif isinstance(estimator, BaseRegressor):
        label_type = "Regression"
    elif (
        isinstance(estimator, BaseAnomalyDetector)
        or isinstance(estimator, BaseSegmenter)
        or isinstance(estimator, BaseSeriesTransformer)
    ):
        label_type = "NoLabel"
    else:
        raise ValueError(f"Unknown estimator type: {type(estimator)}")

    return label_type
