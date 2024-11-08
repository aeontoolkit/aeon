"""Whole-series anomaly detection methods."""

__all__ = [
    "BaseCollectionAnomalyDetector",
    "OutlierDetectionClassifier",
]

from aeon.anomaly_detection.whole_series._outlier_detection import (
    OutlierDetectionClassifier,
)
from aeon.anomaly_detection.whole_series.base import BaseCollectionAnomalyDetector
