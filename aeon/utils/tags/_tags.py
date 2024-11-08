"""Register of estimator tags.

Each dictionary item corresponds to a tag with the key as its name, the contrained
sub-dictionary has the following items:
    class : identifier for the base class of objects this tag applies to
    type : expected type of the tag value. Should be one or a list of:
                "bool" - valid values are True/False
                "str" - valid values are all strings
                "list" - valid values are all lists of arbitrary elements
                ("str", list_of_string) - any string in list_of_string is valid
                ("list", list_of_string) - any sub-list is valid
                ("list||str", list_of_string) - combination of the above
                None - no value for the tag
    description : plain English description of the tag
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["ESTIMATOR_TAGS"]

from aeon.utils import COLLECTIONS_DATA_TYPES, SERIES_DATA_TYPES

ESTIMATOR_TAGS = {
    # all estimators
    "python_version": {
        "class": "estimator",
        "type": ["str", None],
        "description": "Python version specifier (PEP 440) for estimator as str. "
        "e.g. '>=3.6', '>=3.7, <3.9'. If None, no restriction.",
    },
    "python_dependencies": {
        "class": "estimator",
        "type": ["list", "str", None],
        "description": "Python dependency version specifier (PEP 440) for estimator "
        "as str or list of str. e.g. 'tsfresh>=0.20.0', "
        "'[tsfresh>=0.20.0, pandas<2.0.0]'. If None, no restriction.",
    },
    "non_deterministic": {
        "class": "estimator",
        "type": "bool",
        "description": "The estimator is non-deterministic, and multiple runs will "
        "not produce the same output even if a `random_state` is set.",
    },
    "cant_pickle": {
        "class": "estimator",
        "type": "bool",
        "description": "The estimator cannot be pickled.",
    },
    "X_inner_type": {
        "class": "estimator",
        "type": [
            ("list||str", COLLECTIONS_DATA_TYPES + SERIES_DATA_TYPES),
        ],
        "description": "What data structure(s) the estimator uses internally for "
        "fit/predict.",
    },
    "algorithm_type": {
        "class": "estimator",
        "type": [
            (
                "str",
                [
                    "dictionary",
                    "distance",
                    "feature",
                    "hybrid",
                    "interval",
                    "convolution",
                    "shapelet",
                    "deeplearning",
                ],
            ),
            None,
        ],
        "description": "Which type the estimator falls under in its relevant taxonomy"
        " of time series machine learning algorithms.",
    },
    "fit_is_empty": {
        "class": "estimator",
        "type": "bool",
        "description": "fit contains no logic and can be skipped? Yes=True, No=False",
    },
    # capabilities - flags for what an estimator can and cannot do
    "capability:univariate": {
        "class": "estimator",
        "type": "bool",
        "description": "Can the estimator deal with single series input?",
    },
    "capability:multivariate": {
        "class": "estimator",
        "type": "bool",
        "description": "Can the estimator deal with series with multiple channels?",
    },
    "capability:missing_values": {
        "class": "estimator",
        "type": "bool",
        "description": "Can the estimator handle missing data (NA, np.nan) in inputs?",
    },
    "capability:unequal_length": {
        "class": "collection-estimator",
        "type": "bool",
        "description": "Can the estimator handle unequal length time series input?",
    },
    "capability:multithreading": {
        "class": "estimator",
        "type": "bool",
        "description": "Can the estimator set `n_jobs` to use multiple threads?",
    },
    "capability:train_estimate": {
        "class": ["classifier", "regressor"],
        "type": "bool",
        "description": "Can the estimator use an in-built mechanism to estimate its "
        "performance on the training set?",
    },
    "capability:contractable": {
        "class": ["classifier", "regressor"],
        "type": "bool",
        "description": "Can the estimator limiting max fit time?",
    },
    "capability:inverse_transform": {
        "class": "transformer",
        "type": "bool",
        "description": "Can the transformer carrying out an inverse transform?",
    },
    # other
    "returns_dense": {
        "class": "segmenter",
        "type": "bool",
        "description": "Does the segmenter return a list of change points/start index "
        "each segmenter (dense format) or a list indicating which segment each time "
        "point belongs to.",
    },
    "requires_y": {
        "class": [
            "transformer",
            "anomaly-detector",
            "collection_anomaly_detector",
            "segmenter",
        ],
        "type": "bool",
        "description": "Does this estimator require y to be passed in its methods?",
    },
    "removes_unequal_length": {
        "class": "collection-transformer",
        "type": "bool",
        "description": "Is the transformer result guaranteed to be equal length series "
        "or tabular?",
    },
    "removes_missing_values": {
        "class": "transformer",
        "type": "bool",
        "description": "Is the transformer result guaranteed to have no missing "
        "values?",
    },
    "input_data_type": {
        "class": "transformer",
        "type": ("str", ["Series", "Collection"]),
        "description": "The input abstract data type of the transformer, input X. "
        "Series indicates a single series input, Collection indicates a collection of "
        "time series.",
    },
    "output_data_type": {
        "class": "transformer",
        "type": ("str", ["Tabular", "Series", "Collection"]),
        "description": "The output abstract data type of the transformer output, the "
        "transformed X. Tabular indicates 2D output where rows are cases and "
        "unordered attributes are columns. Series indicates a single series output "
        "and collection indicates output is a collection of time series.",
    },
}
