"""Mock estimators for testing and debugging."""

__all__ = [
    # anomaly detection
    "MockAnomalyDetector",
    "MockAnomalyDetectorRequiresFit",
    "MockAnomalyDetectorRequiresY",
    # classification
    "MockClassifier",
    "MockClassifierPredictProba",
    "MockClassifierFullTags",
    "MockClassifierParams",
    "MockClassifierComposite",
    # clustering
    "MockCluster",
    "MockDeepClusterer",
    # collection transformation
    "MockCollectionTransformer",
    # forecasting
    "MockForecaster",
    # regression
    "MockRegressor",
    "MockRegressorFullTags",
    # segmentation
    "MockSegmenter",
    "MockSegmenterRequiresY",
    # series transformation
    "MockSeriesTransformer",
    "MockUnivariateSeriesTransformer",
    "MockMultivariateSeriesTransformer",
    "MockSeriesTransformerNoFit",
]

from aeon.testing.mock_estimators._mock_anomaly_detectors import (
    MockAnomalyDetector,
    MockAnomalyDetectorRequiresFit,
    MockAnomalyDetectorRequiresY,
)
from aeon.testing.mock_estimators._mock_classifiers import (
    MockClassifier,
    MockClassifierComposite,
    MockClassifierFullTags,
    MockClassifierParams,
    MockClassifierPredictProba,
)
from aeon.testing.mock_estimators._mock_clusterers import MockCluster, MockDeepClusterer
from aeon.testing.mock_estimators._mock_collection_transformers import (
    MockCollectionTransformer,
)
from aeon.testing.mock_estimators._mock_forecasters import MockForecaster
from aeon.testing.mock_estimators._mock_regressors import (
    MockRegressor,
    MockRegressorFullTags,
)
from aeon.testing.mock_estimators._mock_segmenters import (
    MockSegmenter,
    MockSegmenterRequiresY,
)
from aeon.testing.mock_estimators._mock_series_transformers import (
    MockMultivariateSeriesTransformer,
    MockSeriesTransformer,
    MockSeriesTransformerNoFit,
    MockUnivariateSeriesTransformer,
)
