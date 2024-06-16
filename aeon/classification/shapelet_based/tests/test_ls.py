"""Learned Shapelets tests."""

import numpy as np
import pytest

from aeon.classification.shapelet_based import LearningShapeletClassifier
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tslearn", "tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_get_transform():
    """Learned Shapelets tests."""
    X = make_example_3d_numpy(return_y=False, n_cases=10)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    clf = LearningShapeletClassifier(max_iter=100)
    clf.fit(X, y)
    with pytest.raises(ValueError):
        clf.get_transform(X)
    with pytest.raises(ValueError):
        clf.get_locations(X)
    clf = LearningShapeletClassifier(max_iter=100, save_transformed_data=True)
    with pytest.raises(ValueError):
        clf.get_transform(X)
    with pytest.raises(ValueError):
        clf.get_locations(X)
    clf.fit(X, y)
    t = clf.get_transform(X)
    assert isinstance(t, np.ndarray)
