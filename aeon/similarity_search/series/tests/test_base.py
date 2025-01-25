"""Test for series similarity search base class."""

__maintainer__ = ["baraline"]

import pytest

from aeon.testing.mock_estimators._mock_similarity_searchers import (
    MockSeriesSimilaritySearch,
)
from aeon.testing.testing_data import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)


def test_input_shape_fit_predict_series():
    """Test input shapes."""
    estimator = MockSeriesSimilaritySearch()
    # dummy data to pass to fit when testing predict/predict_proba
    X_3D_uni = make_example_3d_numpy(n_channels=1, return_y=False)
    X_3D_multi = make_example_3d_numpy(n_channels=2, return_y=False)
    X_3D_uni_list = make_example_3d_numpy_list(n_channels=1, return_y=False)
    X_3D_multi_list = make_example_3d_numpy_list(n_channels=2, return_y=False)
    X_2D_uni = make_example_2d_numpy_series(n_channels=1)
    X_2D_multi = make_example_2d_numpy_series(n_channels=2)
    X_1D = make_example_1d_numpy()

    valid_inputs_fit = [X_1D, X_2D_uni, X_2D_multi]
    # 1D is converted to 2D univariate
    for _input in valid_inputs_fit:
        estimator.fit(_input)

    invalid_inputs_fit = [
        X_3D_multi,
        X_3D_uni,
        X_3D_multi_list,
        X_3D_uni_list,
    ]
    for _input in invalid_inputs_fit:
        with pytest.raises(ValueError):
            estimator.fit(_input)

    estimator_multi = MockSeriesSimilaritySearch().fit(X_2D_multi)
    estimator_uni = MockSeriesSimilaritySearch().fit(X_2D_uni)

    estimator_uni.predict(X_2D_uni)
    # 1D is converted to 2D univariate
    estimator_uni.predict(X_1D)
    estimator_multi.predict(X_2D_multi)

    with pytest.raises(ValueError):
        estimator_uni.predict(X_2D_multi)
    with pytest.raises(ValueError):
        estimator_multi.predict(X_2D_uni)

    for _input in [X_3D_uni, X_3D_uni_list]:
        with pytest.raises(ValueError):
            estimator_uni.predict(_input)

    for _input in [X_3D_multi, X_3D_multi_list]:
        with pytest.raises(ValueError):
            estimator_multi.predict(_input)
