"""Register of estimator base classes.

BASE_CLASS_REGISTER - dictionary
    Dictionary of base classes for each estimator type. Keys are identifier strings,
    values are base classes.

VALID_ESTIMATOR_BASES - dictionary
    Dictionary of base classes that are valid for estimators to inherit from. Subset of
    BASE_CLASS_REGISTER. Keys are identifier strings, values are base classes.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "BASE_CLASS_REGISTER",
    "VALID_ESTIMATOR_BASES",
]


from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.anomaly_detection.whole_series.base import BaseCollectionAnomalyDetector
from aeon.base import BaseAeonEstimator, BaseCollectionEstimator, BaseSeriesEstimator
from aeon.classification.base import BaseClassifier
from aeon.classification.early_classification import BaseEarlyClassifier
from aeon.clustering.base import BaseClusterer
from aeon.regression.base import BaseRegressor
from aeon.segmentation.base import BaseSegmenter
from aeon.similarity_search.base import BaseSimilaritySearch
from aeon.transformations.base import BaseTransformer
from aeon.transformations.collection import BaseCollectionTransformer
from aeon.transformations.series import BaseSeriesTransformer

# all base classes
BASE_CLASS_REGISTER = {
    # abstract - no estimator directly inherits from these
    "collection-estimator": BaseCollectionEstimator,
    "estimator": BaseAeonEstimator,
    "series-estimator": BaseSeriesEstimator,
    "transformer": BaseTransformer,
    # estimator types
    "anomaly-detector": BaseAnomalyDetector,
    "collection_anomaly_detector": BaseCollectionAnomalyDetector,
    "collection-transformer": BaseCollectionTransformer,
    "classifier": BaseClassifier,
    "clusterer": BaseClusterer,
    "early_classifier": BaseEarlyClassifier,
    "regressor": BaseRegressor,
    "segmenter": BaseSegmenter,
    "similarity_searcher": BaseSimilaritySearch,
    "series-transformer": BaseSeriesTransformer,
}

# base classes which are valid for estimator to directly inherit from
VALID_ESTIMATOR_BASES = {
    k: BASE_CLASS_REGISTER[k]
    for k in BASE_CLASS_REGISTER.keys()
    - {"estimator", "collection-estimator", "series-estimator", "transformer"}
}
