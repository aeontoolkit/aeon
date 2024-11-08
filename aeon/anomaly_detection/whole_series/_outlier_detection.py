"""Basic outlier detection classifier."""

from sklearn.ensemble import IsolationForest

from aeon.anomaly_detection.whole_series.base import BaseCollectionAnomalyDetector
from aeon.base._base import _clone_estimator


class OutlierDetectionClassifier(BaseCollectionAnomalyDetector):
    """Basic outlier detection classifier."""

    _tags = {
        "X_inner_type": "numpy2D",
    }

    def __init__(self, estimator, random_state=None):
        self.estimator = estimator
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y=None):
        self.estimator_ = _clone_estimator(
            self.estimator, random_state=self.random_state
        )
        self.estimator_.fit(X, y)
        return self

    def _predict(self, X):
        pred = self.estimator_.predict(X)
        pred[pred == -1] = 0
        return pred

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {"estimator": IsolationForest(n_estimators=3)}
