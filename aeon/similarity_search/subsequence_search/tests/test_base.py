"""Test for subsequence search base class."""

import pytest

from aeon.testing.mock_estimators._mock_similarity_searchers import (
    MockMatrixProfile,
    MockSubsequenceSearch,
)
from aeon.testing.testing_data import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)

BASES = [MockMatrixProfile, MockSubsequenceSearch]


@pytest.mark.parametrize("base", BASES)
def test_input_shape_fit_neighbord_motifs(base):
    """Test input shapes."""
    estimator = base()
    # dummy data to pass to fit when testing predict/predict_proba
    X_3D_uni = make_example_3d_numpy(n_channels=1, return_y=False)
    X_3D_multi = make_example_3d_numpy(n_channels=2, return_y=False)
    X_3D_uni_list = make_example_3d_numpy_list(n_channels=1, return_y=False)
    X_3D_multi_list = make_example_3d_numpy_list(n_channels=2, return_y=False)
    X_2D_uni = make_example_2d_numpy_series(n_channels=1)
    X_2D_multi = make_example_2d_numpy_series(n_channels=2)
    X_1D = make_example_1d_numpy()

    valid_inputs_fit = [X_3D_uni, X_3D_multi, X_3D_uni_list, X_3D_multi_list]
    # Valid inputs
    for _input in valid_inputs_fit:
        estimator.fit(_input)

    invalid_inputs_fit = [X_2D_uni, X_2D_multi, X_1D]
    for _input in invalid_inputs_fit:
        with pytest.raises(TypeError):
            estimator.fit(_input)

    valid_inputs_neighboord_motifs_uni = [X_2D_uni]
    invalid_inputs_neighboord_motifs_uni = [
        X_1D,
        X_3D_uni,
        X_3D_uni_list,
    ]
    invalid_inputs_neighboord_motifs_multi = [
        X_3D_multi,
        X_3D_multi_list,
    ]
    L = 5
    estimator_multi = base(length=L).fit(X_3D_multi)
    estimator_uni = base(length=L).fit(X_3D_uni)

    for _input in valid_inputs_neighboord_motifs_uni:
        estimator_uni.find_neighbors(_input[:, :L])
        estimator_uni.find_motifs(_input)
        with pytest.raises(ValueError):
            # Wrong number of channels
            estimator_multi.find_neighbors(_input)
        with pytest.raises(ValueError):
            estimator_multi.find_motifs(_input)
        # X length not of size L
        with pytest.raises(ValueError):
            estimator_uni.find_neighbors(_input[:, : L + 2])

    for _input in invalid_inputs_neighboord_motifs_uni:
        with pytest.raises(TypeError):
            estimator_uni.find_neighbors(_input)
        with pytest.raises(TypeError):
            estimator_uni.find_motifs(_input)
        with pytest.raises(TypeError):
            estimator_multi.find_neighbors(_input)
        with pytest.raises(TypeError):
            estimator_multi.find_motifs(_input)

    for _input in invalid_inputs_neighboord_motifs_multi:
        with pytest.raises(TypeError):
            estimator_uni.find_neighbors(_input)
        with pytest.raises(TypeError):
            estimator_uni.find_motifs(_input)
        with pytest.raises(TypeError):
            estimator_multi.find_neighbors(_input)
        with pytest.raises(TypeError):
            estimator_multi.find_motifs(_input)
