"""Test for collection similarity search base class."""

__maintainer__ = ["baraline"]

import pytest

from aeon.testing.mock_estimators._mock_similarity_searchers import (
    MockCollectionSimilaritySearch,
)
from aeon.testing.testing_data import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
    make_example_3d_numpy,
)


def test_input_shape_fit_predict_collection():
    """Test input shapes."""
    estimator = MockCollectionSimilaritySearch()
    # dummy data to pass to fit when testing predict/predict_proba
    X_3D_uni = make_example_3d_numpy(n_channels=1, return_y=False)
    X_3D_multi = make_example_3d_numpy(n_channels=2, return_y=False)
    X_2D_uni = make_example_2d_numpy_series(n_channels=1)
    X_2D_multi = make_example_2d_numpy_series(n_channels=2)
    X_1D = make_example_1d_numpy()

    # 2D are converted to 3D
    valid_inputs_fit = [
        X_3D_uni,
        X_3D_multi,
        X_2D_uni,
        X_2D_multi,
    ]
    # Valid inputs
    for _input in valid_inputs_fit:
        estimator.fit(_input)

    with pytest.raises(ValueError):
        estimator.fit(X_1D)

    estimator_multi = MockCollectionSimilaritySearch().fit(X_3D_multi)
    estimator_uni = MockCollectionSimilaritySearch().fit(X_3D_uni)

    estimator_uni.predict(X_2D_uni)
    estimator_multi.predict(X_2D_multi)

    with pytest.raises(ValueError):
        estimator_uni.predict(X_2D_multi)
    with pytest.raises(ValueError):
        estimator_multi.predict(X_2D_uni)
    with pytest.raises(TypeError):
        estimator_uni.predict(X_3D_uni)
    with pytest.raises(TypeError):
        estimator_multi.predict(X_3D_multi)
