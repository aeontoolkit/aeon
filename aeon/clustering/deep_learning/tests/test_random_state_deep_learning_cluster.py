"""Unit tests for clusterer deep learning random_state functionality."""

import inspect

import numpy as np
import pytest

from aeon.clustering import deep_learning
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies

__maintainer__ = ["hadifawaz1999"]

_deep_clr_classes = [
    member[1] for member in inspect.getmembers(deep_learning, inspect.isclass)
]


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("deep_clr", _deep_clr_classes)
def test_random_state_deep_learning_clr(deep_clr):
    """Test Deep Clusterer seeding."""
    if not (deep_clr.__name__ in ["BaseDeepClusterer"]):
        random_state = 42

        X, _ = make_example_3d_numpy(random_state=random_state)

        test_params = deep_clr.get_test_params()[0]
        test_params["random_state"] = random_state

        deep_clr1 = deep_clr(**test_params)
        deep_clr1.fit(X)

        encoder1 = deep_clr1.training_model_.layers[1]
        decoder1 = deep_clr1.training_model_.layers[2]
        encoder_layers1 = encoder1.layers[1:]
        decoder_layers1 = decoder1.layers[1:]

        deep_clr2 = deep_clr(**test_params)
        deep_clr2.fit(X)

        encoder2 = deep_clr2.training_model_.layers[1]
        decoder2 = deep_clr2.training_model_.layers[2]
        encoder_layers2 = encoder2.layers[1:]
        decoder_layers2 = decoder2.layers[1:]

        # test encoders
        for i in range(len(encoder_layers1)):
            weights1 = encoder_layers1[i].get_weights()
            weights2 = encoder_layers2[i].get_weights()

            assert len(weights1) == len(weights2)

            for j in range(len(weights1)):
                _weight1 = np.asarray(weights1[j])
                _weight2 = np.asarray(weights2[j])

                np.testing.assert_almost_equal(_weight1, _weight2, 4)

        # test decoders
        for i in range(len(decoder_layers1)):
            weights1 = decoder_layers1[i].get_weights()
            weights2 = decoder_layers2[i].get_weights()

            assert len(weights1) == len(weights2)

            for j in range(len(weights1)):
                _weight1 = np.asarray(weights1[j])
                _weight2 = np.asarray(weights2[j])

                np.testing.assert_almost_equal(_weight1, _weight2, 4)
