# -*- coding: utf-8 -*-
"""Unit tests for regressors deep learning base class functionality."""

import os

import pytest

from aeon.regression.deep_learning.base import BaseDeepRegressor
from aeon.utils.validation._dependencies import _check_soft_dependencies

__author__ = ["achieveordie", "hadifawaz1999"]


class _DummyDeepRegressor(BaseDeepRegressor):
    """Dummy Deep Regressor for testing empty base deep class save utilities."""

    def __init__(self):
        super(_DummyDeepRegressor, self).__init__()

    def build_model(self, input_shape, n_classes, **kwargs):
        import tensorflow as tf

        input_layer = tf.keras.layers.Input(input_shape)
        output_layer = tf.keras.layers.Dense(units=n_classes)(input_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile()

        return model

    def _fit(self, X, y):
        X = X.transpose(0, 2, 1)

        self.input_shape_ = X.shape[1:]
        self.model_ = self.build_model(self.input_shape_)

        self.history = self.model_.fit(
            X,
            y,
            batch_size=16,
            epochs=2,
        )

        return self


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_dummy_deep_regressor():
    import numpy as np

    # create a dummy regressor
    dummy_deep_rg = _DummyDeepRegressor()

    # test fit function on random data
    dummy_deep_rg.fit(
        X=np.random.normal(size=(10, 100)), y=np.random.normal(size=(10,))
    )

    # test save last model to file than delete it

    dummy_deep_rg.save_last_model_to_file()

    os.remove("./last_model.hdf5")

    # test summary of model

    dummy_deep_rg.summary()
