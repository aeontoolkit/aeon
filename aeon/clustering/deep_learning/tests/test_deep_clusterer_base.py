"""Unit tests for clusterer deep learning base class functionality."""

import tempfile

import numpy as np
import pytest

from aeon.clustering._dummy import DummyClusterer
from aeon.testing.data_generation import make_example_2d_numpy_collection
from aeon.testing.mock_estimators import MockDeepClusterer
from aeon.utils.validation._dependencies import _check_soft_dependencies

__maintainer__ = ["hadifawaz1999"]


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("estimator", [None, DummyClusterer(n_clusters=2)])
def test_base_deep_clusterer(estimator):
    """Test base deep clusterer."""
    with tempfile.TemporaryDirectory() as tmp:
        last_file_name = "temp"
        # create a dummy deep classifier
        dummy_deep_clr = MockDeepClusterer(
            estimator=estimator, last_file_name=last_file_name
        )
        # generate random data
        X, y = make_example_2d_numpy_collection()
        # test fit function on random data
        dummy_deep_clr.fit(X=X)
        # test save last model to file than delete it
        dummy_deep_clr.save_last_model_to_file(file_path=tmp)

        # test summary of model
        assert dummy_deep_clr.summary() is not None
        ypred = dummy_deep_clr.predict(X)
        assert ypred is not None
        assert len(ypred) == len(y)
        ypred_proba = dummy_deep_clr.predict_proba(X)
        assert ypred_proba is not None
        assert len(ypred_proba[0]) == len(np.unique(y))
