"""Wrapper for imblearn minority class rebalancer SMOTE."""

from imblearn.over_sampling import SMOTE as smote
import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer

__maintainer__ = ["TonyBagnall"]
__all__ = ["SMOTE"]


class SMOTE(BaseCollectionTransformer):
    """Wrapper for SMOTE transform."""

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "requires_y": True,
    }

    def __init__(self, sampling_strategy="auto", random_state=None, k_neighbors=5):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        super().__init__()

    def _transform(self, X, y=None):
        return self

    def _fit_transform(self, X, y=None):

        self.smote_ = smote(sampling_strategy=self.sampling_strategy,
                            random_state=self.random_state, k_neighbors=self.k_neighbors)
        res_X, res_y = self.smote_.fit_resample(np.squeeze(X), y)
        return res_X, res_y


if __name__ == "__main__":
    # Example usage
    import numpy as np

    X = np.random.rand(100, 1, 10)
    y = np.random.randint(0, 2, 100)

    transformer = SMOTE()
    res_X, res_y = transformer.fit_transform(X, y)
    print(res_X.shape, res_y.shape)
    # Expected output: (200, 3, 10) (200,)
