"""Datasets used for testing."""

import numpy as np

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.base import BaseCollectionEstimator, BaseSeriesEstimator
from aeon.classification import BaseClassifier
from aeon.classification.early_classification import BaseEarlyClassifier
from aeon.clustering import BaseClusterer
from aeon.forecasting import BaseForecaster
from aeon.regression import BaseRegressor
from aeon.segmentation import BaseSegmenter
from aeon.similarity_search.collection import BaseCollectionSimilaritySearch
from aeon.similarity_search.series import BaseSeriesSimilaritySearch
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
from aeon.utils.conversion import convert_collection

data_rng = np.random.RandomState(42)

# Collection testing data

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

EQUAL_LENGTH_UNIVARIATE_COLLETION_SIMILARITY_SEARCH = {
    "numpy3D": {
        "train": (
            make_example_3d_numpy(
                n_cases=10,
                n_channels=1,
                n_timepoints=20,
                random_state=data_rng.randint(np.iinfo(np.int32).max),
                return_y=False,
            ),
            None,
        ),
        "test": (
            make_example_2d_numpy_series(
                n_timepoints=10,
                n_channels=1,
                random_state=data_rng.randint(np.iinfo(np.int32).max),
            ),
            None,
        ),
    },
    "np-list": {
        "train": (
            make_example_3d_numpy_list(
                n_cases=10,
                n_channels=1,
                min_n_timepoints=20,
                max_n_timepoints=20,
                random_state=data_rng.randint(np.iinfo(np.int32).max),
                return_y=False,
            ),
            None,
        ),
        "test": (
            make_example_2d_numpy_series(
                n_timepoints=10,
                n_channels=1,
                random_state=data_rng.randint(np.iinfo(np.int32).max),
            ),
            None,
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

EQUAL_LENGTH_MULTIVARIATE_COLLETION_SIMILARITY_SEARCH = {
    "numpy3D": {
        "train": (
            make_example_3d_numpy(
                n_cases=10,
                n_channels=2,
                n_timepoints=20,
                random_state=data_rng.randint(np.iinfo(np.int32).max),
                return_y=False,
            ),
            None,
        ),
        "test": (
            make_example_2d_numpy_series(
                n_timepoints=10,
                n_channels=2,
                random_state=data_rng.randint(np.iinfo(np.int32).max),
            ),
            None,
        ),
    },
    "np-list": {
        "train": (
            make_example_3d_numpy_list(
                n_cases=10,
                n_channels=2,
                min_n_timepoints=20,
                max_n_timepoints=20,
                random_state=data_rng.randint(np.iinfo(np.int32).max),
                return_y=False,
            ),
            None,
        ),
        "test": (
            make_example_2d_numpy_series(
                n_timepoints=10,
                n_channels=2,
                random_state=data_rng.randint(np.iinfo(np.int32).max),
            ),
            None,
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

UNEQUAL_LENGTH_UNIVARIATE_COLLETION_SIMILARITY_SEARCH = {
    "np-list": {
        "train": (
            make_example_3d_numpy_list(
                n_cases=10,
                n_channels=1,
                min_n_timepoints=10,
                max_n_timepoints=20,
                random_state=data_rng.randint(np.iinfo(np.int32).max),
                return_y=False,
            ),
            None,
        ),
        "test": (
            make_example_2d_numpy_series(
                n_timepoints=10,
                n_channels=1,
                random_state=data_rng.randint(np.iinfo(np.int32).max),
            ),
            None,
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
    },
    "np-list": {
        "train": (
            convert_collection(X_classification_missing_train, "np-list"),
            y_classification_missing_train,
        ),
        "test": (
            convert_collection(X_classification_missing_test, "np-list"),
            y_classification_missing_test,
        ),
    },
}

X_regression_missing_train, y_regression_missing_train = make_example_3d_numpy(
    n_cases=10,
    n_channels=1,
    n_timepoints=20,
    random_state=data_rng.randint(np.iinfo(np.int32).max),
    regression_target=True,
)
X_regression_missing_test, y_regression_missing_test = make_example_3d_numpy(
    n_cases=5,
    n_channels=1,
    n_timepoints=20,
    random_state=data_rng.randint(np.iinfo(np.int32).max),
    regression_target=True,
)
X_regression_missing_train[:, :, data_rng.choice(20, 2)] = np.nan
X_regression_missing_test[:, :, data_rng.choice(20, 2)] = np.nan

MISSING_VALUES_REGRESSION = {
    "numpy3D": {
        "train": (X_regression_missing_train, y_regression_missing_train),
        "test": (X_regression_missing_test, y_regression_missing_test),
    },
    "np-list": {
        "train": (
            convert_collection(X_regression_missing_train, "np-list"),
            y_regression_missing_train,
        ),
        "test": (
            convert_collection(X_regression_missing_test, "np-list"),
            y_regression_missing_test,
        ),
    },
}

# Series testing data

X_series = make_example_1d_numpy(
    n_timepoints=40, random_state=data_rng.randint(np.iinfo(np.int32).max)
)
X_series2 = X_series[20:40]
X_series = X_series[:20]
UNIVARIATE_SERIES_NONE = {"train": (X_series, None), "test": (X_series2, None)}

X_series_mv = make_example_2d_numpy_series(
    n_timepoints=40,
    n_channels=2,
    axis=1,
    random_state=data_rng.randint(np.iinfo(np.int32).max),
)
X_series_mv2 = X_series_mv[:, 20:40]
X_series_mv = X_series_mv[:, :20]
MULTIVARIATE_SERIES_NONE = {
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
MISSING_VALUES_SERIES_NONE = {
    "train": (X_series_mi, None),
    "test": (X_series_mi2, None),
}

# All testing data

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
        f"EqualLengthUnivariate-SimilaritySearch-{k}": v
        for k, v in EQUAL_LENGTH_UNIVARIATE_COLLETION_SIMILARITY_SEARCH.items()
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
        f"EqualLengthMultivariate-SimilaritySearch-{k}": v
        for k, v in EQUAL_LENGTH_MULTIVARIATE_COLLETION_SIMILARITY_SEARCH.items()
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
        f"UnequalLengthUnivariate-CollectionSimilaritySearch-{k}": v
        for k, v in UNEQUAL_LENGTH_UNIVARIATE_COLLETION_SIMILARITY_SEARCH.items()
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
FULL_TEST_DATA_DICT.update({"UnivariateSeries-None": UNIVARIATE_SERIES_NONE})
FULL_TEST_DATA_DICT.update({"MultivariateSeries-None": MULTIVARIATE_SERIES_NONE})
FULL_TEST_DATA_DICT.update({"MissingValues-None": MISSING_VALUES_SERIES_NONE})


def _get_datatypes_for_estimator(estimator):
    """Get all data types for estimator.

    Parameters
    ----------
    estimator : BaseAeonEstimator instance or class
        Estimator instance or class to check for valid input data types.

    Returns
    -------
    datatypes : list of tuple
        List of valid data types keys for the estimator usable in
        FULL_TEST_DATA_DICT. Each tuple is formatted (data_key, label_key).
    """
    datatypes = []
    univariate, multivariate, unequal_length, missing_values = (
        _get_capabilities_for_estimator(estimator)
    )
    task = _get_task_for_estimator(estimator)

    inner_types = estimator.get_tag("X_inner_type")
    if not isinstance(inner_types, list):
        inner_types = [inner_types]

    if isinstance(estimator, BaseCollectionEstimator):
        for inner_type in inner_types:
            if univariate:
                s = f"EqualLengthUnivariate-{task}-{inner_type}"
                if s in FULL_TEST_DATA_DICT:
                    datatypes.append(s)

                if unequal_length:
                    s = f"UnequalLengthUnivariate-{task}-{inner_type}"
                    if s in FULL_TEST_DATA_DICT:
                        datatypes.append(s)

            if multivariate:
                s = f"EqualLengthMultivariate-{task}-{inner_type}"
                if s in FULL_TEST_DATA_DICT:
                    datatypes.append(s)

                if unequal_length:
                    s = f"UnequalLengthMultivariate-{task}-{inner_type}"
                    if s in FULL_TEST_DATA_DICT:
                        datatypes.append(s)

        if missing_values:
            datatypes.append(f"MissingValues-{task}-numpy3D")
    elif isinstance(estimator, BaseSeriesEstimator):
        if univariate:
            datatypes.append(f"UnivariateSeries-{task}")
        if multivariate:
            datatypes.append(f"MultivariateSeries-{task}")
        if missing_values:
            datatypes.append(f"MissingValues-{task}")
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


def _get_task_for_estimator(estimator):
    """Get task string used to select the correct test data for the estimator.

    Parameters
    ----------
    estimator : BaseAeonEstimator instance or class
        Estimator instance or class to find the task string for.

    Returns
    -------
    data_label : str
        Task string for the estimator used in forming a key from FULL_TEST_DATA_DICT.
    """
    # collection data with class labels
    if (
        isinstance(estimator, BaseClassifier)
        or isinstance(estimator, BaseEarlyClassifier)
        or isinstance(estimator, BaseClusterer)
        or isinstance(estimator, BaseCollectionTransformer)
    ):
        data_label = "Classification"
    # collection data with continuous target labels
    elif isinstance(estimator, BaseRegressor):
        data_label = "Regression"
    elif isinstance(estimator, BaseCollectionSimilaritySearch):
        data_label = "CollectionSimilaritySearch"
    # series data with no secondary input
    elif (
        isinstance(estimator, BaseAnomalyDetector)
        or isinstance(estimator, BaseSegmenter)
        or isinstance(estimator, BaseSeriesTransformer)
        or isinstance(estimator, BaseForecaster)
        or isinstance(estimator, BaseSeriesSimilaritySearch)
    ):
        data_label = "None"
    else:
        raise ValueError(f"Unknown estimator type: {type(estimator)}")

    return data_label
